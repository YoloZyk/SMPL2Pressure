import sys
import smplx
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset, DataLoader

sys.path.append('/workspace/zyk/SMPL2Pressure')

# from datareader import MyMoYoDataset, PressurePoseDataset, TIPDataset
from lib.utils.mesh_utils import HDfier
from lib.utils.part_volumes import PartVolume
from lib.utils.static import SMPL_PART_BOUNDS, FID_TO_PART, PART_VID_FID, HD_SMPL_MAP


def cal_cop(pressure, resolution=0.7874):
    B, Y, X = pressure.shape
    
    x_coords = torch.arange(X, device=pressure.device)  # [0, 1, 2, ..., 36]
    y_coords = torch.arange(Y, device=pressure.device)  # [0, 1, 2, ..., 109]
    
    total_pressure = pressure.sum(dim=(1, 2))  # shape: (B,)
    total_pressure = torch.where(total_pressure == 0, torch.tensor(1e-8, device=pressure.device), total_pressure)
    
    x_weighted_sum = (pressure * x_coords).sum(dim=(1, 2))  # shape: (B,)
    y_weighted_sum = (pressure * y_coords.view(-1, 1)).sum(dim=(1, 2))  # shape: (B,)
    
    x_centers = x_weighted_sum / total_pressure  # shape: (B,)
    y_centers = y_weighted_sum / total_pressure  # shape: (B,)
    
    cop_x = 0.01 * (x_centers - 18) / resolution
    cop_y = 0.01 * (86 - y_centers) / resolution

    cop_xy = torch.stack([cop_x, cop_y, torch.zeros_like(cop_x)], dim=1)
    cop_sensor_xy = torch.stack([x_centers, y_centers], dim=1)
    
    return cop_xy, cop_sensor_xy
    

class StabilityLossCoP(nn.Module):
    def __init__(self,
                 faces,
                 cop_w=10,
                 cop_k=100,
                 contact_thresh=0.1,
                 model_type='smpl',
                 device=torch.device('cpu'),
                 ):
        super().__init__()
        """
        Loss that ensures that the COM of the SMPL mesh is close to the center of support 
        """
        if model_type == 'smpl':
            num_faces = 13776
            num_verts_hd = 20000

        assert faces is not None, 'Faces tensor is none'
        if type(faces) is not torch.Tensor:
            faces = torch.tensor(faces.astype(np.int64), dtype=torch.long).to(device)
        self.register_buffer('faces', faces)

        self.cop_w = cop_w
        self.cop_k = cop_k
        self.contact_thresh = contact_thresh

        self.hdfy_op = HDfier(model_type=model_type)

        with open(SMPL_PART_BOUNDS, 'rb') as f:
            d = pkl.load(f)
            self.part_bounds = {k: d[k] for k in sorted(d)}
        self.part_order = sorted(self.part_bounds)

        with open(PART_VID_FID, 'rb') as f:
            self.part_vid_fid = pkl.load(f)

        # mapping between vid_hd and fid
        with open(HD_SMPL_MAP, 'rb') as f:
            faces_vert_is_sampled_from = pkl.load(f)['faces_vert_is_sampled_from']
        index_row_col = torch.stack(
            [torch.LongTensor(np.arange(0, num_verts_hd)), torch.LongTensor(faces_vert_is_sampled_from)], dim=0)
        values = torch.ones(num_verts_hd, dtype=torch.float)
        size = torch.Size([num_verts_hd, num_faces])
        hd_vert_on_fid = torch.sparse.FloatTensor(index_row_col, values, size)

        # mapping between fid and part label
        with open(FID_TO_PART, 'rb') as f:
            fid_to_part_dict = pkl.load(f)
        fid_to_part = torch.zeros([len(fid_to_part_dict.keys()), len(self.part_order)], dtype=torch.float32)
        for fid, partname in fid_to_part_dict.items():
            part_idx = self.part_order.index(partname)
            fid_to_part[fid, part_idx] = 1.

        # mapping between vid_hd and part label
        self.hd_vid_in_part = self.vertex_id_to_part_mapping(hd_vert_on_fid, fid_to_part)

    def compute_triangle_area(self, triangles):
        ### Compute the area of each triangle in the mesh
        # Compute the cross product of the two vectors of each triangle
        # Then compute the length of the cross product
        # Finally, divide by 2 to get the area of each triangle

        vectors = torch.diff(triangles, dim=2)
        crosses = torch.cross(vectors[:, :, 0], vectors[:, :, 1])
        area = torch.norm(crosses, dim=2) / 2
        return area

    def compute_per_part_volume(self, vertices):
        """
        Compute the volume of each part in the reposed mesh
        """
        part_volume = []
        for part_name, part_bounds in self.part_bounds.items():
            # get part vid and fid
            part_vid = torch.LongTensor(self.part_vid_fid[part_name]['vert_id']).to(vertices.device)
            part_fid = torch.LongTensor(self.part_vid_fid[part_name]['face_id']).to(vertices.device)
            pv = PartVolume(part_name, vertices, self.faces)
            for bound_name, bound_vids in part_bounds.items():
                pv.close_mesh(bound_vids)
            # add extra vids and fids to original part ids
            new_vert_ids = torch.LongTensor(pv.new_vert_ids).to(vertices.device)
            new_face_ids = torch.LongTensor(pv.new_face_ids).to(vertices.device)
            part_vid = torch.cat((part_vid, new_vert_ids), dim=0)
            part_fid = torch.cat((part_fid, new_face_ids), dim=0)
            pv.extract_part_triangles(part_vid, part_fid)
            part_volume.append(pv.part_volume())
        return torch.vstack(part_volume).permute(1, 0).to(vertices.device)

    def vertex_id_to_part_volume_mapping(self, per_part_volume, device):
        batch_size = per_part_volume.shape[0]
        self.hd_vid_in_part = self.hd_vid_in_part.to(device)
        hd_vid_in_part = self.hd_vid_in_part[None, :, :].repeat(batch_size, 1, 1)
        vid_to_vol = torch.bmm(hd_vid_in_part, per_part_volume[:, :, None])
        return vid_to_vol

    def vertex_id_to_part_mapping(self, hd_vert_on_fid, fid_to_part):
        vid_to_part = torch.mm(hd_vert_on_fid, fid_to_part)
        return vid_to_part
    
    def get_com_mat(self, vertices, dataset):
        per_part_volume = self.compute_per_part_volume(vertices)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)
        # get volume per vertex id in the hd mesh
        volume_per_vert_hd = self.vertex_id_to_part_volume_mapping(per_part_volume, vertices.device)
        # calculate com using volume weighted mean
        com = torch.sum(vertices_hd * volume_per_vert_hd, dim=1) / torch.sum(volume_per_vert_hd, dim=1)
        # project com to ground plane (x-y plane)
        if dataset == 'moyo':
            com_mat = torch.stack([com[:, 0] + 18.5/78.74, 86.4/78.74 - com[:, 1]], dim=1) * 100 * 0.7874
        elif dataset == 'pressurepose':
            com_mat = torch.stack([com[:, 0] - 9.3/35, 65.4/35 - com[:, 1]], dim=1) * 100 * 0.35
        elif dataset == 'tip':
            com_mat = torch.stack([(com[:, 0] - 0/48) * 48, (59/33 - com[:, 1]) * 33], dim=1)
        else:
            com_mat = torch.stack([com[:, 0]*0, com[:, 1]]*0, dim=1)
        
        return com_mat.cpu().detach()
    
    def forward(self, vertices, pressure):
        # import pdb; pdb.set_trace()
        # Note: the vertices should be aligned along z-axis and in world coordinates
        batch_size = vertices.shape[0]
        # calculate per part volume
        per_part_volume = self.compute_per_part_volume(vertices)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)
        # get volume per vertex id in the hd mesh
        volume_per_vert_hd = self.vertex_id_to_part_volume_mapping(per_part_volume, vertices.device)
        # calculate com using volume weighted mean
        com = torch.sum(vertices_hd * volume_per_vert_hd, dim=1) / torch.sum(volume_per_vert_hd, dim=1)

        # project com, cop to ground plane (x-y plane)
        # weight loss by number of contact vertices to zero out if zero vertices in contact
        com_xy = torch.stack([com[:, 0], com[:, 1], torch.zeros_like(com)[:, 0]], dim=1)
        # moyo
        # com_mat = torch.stack([com[:, 0] + 18.5/78.74, 86.4/78.74 - com[:, 1]], dim=1) * 100 * 0.7874
        # pressurepose
        # com_mat = torch.stack([com[:, 0] - 9.3/35, 65.4/35 - com[:, 1]], dim=1) * 100 * 0.35
        # tip
        com_mat = torch.stack([(com[:, 0] - 0/48) * 48, (59/33 - com[:, 1]) * 33], dim=1)
        
        cop_xy, cop_mat = cal_cop(pressure)
        
        ground_plane_height = 0.0
        eps = 1e-6
        vertex_height = (vertices_hd[:, :, 1] - ground_plane_height)
        inside_mask = (vertex_height < 0.0).float()
        outside_mask = (vertex_height >= 0.0).float()
        pressure_weights = inside_mask * (1 - self.cop_k * vertex_height) + outside_mask * torch.exp(
            -self.cop_w * vertex_height)
        
        cop = torch.sum(vertices_hd * pressure_weights.unsqueeze(-1), dim=1) / (
                    torch.sum(pressure_weights, dim=1, keepdim=True) + eps)
        # cop_xy_ipman = torch.stack([cop[:, 0], cop[:, 1], torch.zeros_like(cop)[:, 0]], dim=1)
        cop_mat_ipman = torch.stack([cop[:, 0] + 18/78.74, 86/78.74 - cop[:, 1]], dim=1) * 100 * 0.7874
        
        # stability_loss = (contact_confidence * torch.norm(com_xz - contact_centroid_xz, dim=1)).sum(dim=-1)
        stability_loss = (torch.norm(com_xy - cop_xy, dim=1))
        # import pdb; pdb.set_trace()
        return stability_loss.mean(), com_mat, cop_mat_ipman


# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     smpl_model = smplx.create(SMPL_MODEL, model_type='smpl', gender='neutral', ext='pkl')
#     faces = smpl_model.faces
#
#     # train_data = MyMoYoDataset(split='test')
#     # train_data = PressurePoseDataset(split='test')
#     train_data = TIPDataset(split='test')
#
#     train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
#     loss = StabilityLossCoP(faces, device=device)
#     for batch in train_loader:
#         vertices = batch['vertices'].to(device)
#         pressure = batch['pressure'].to(device)
#         cop_xy, cop_mat = cal_cop(pressure)
#         stability_loss, com_mat, cop_ipman = loss(vertices, pressure)
#
#         print(stability_loss)
#         viz_pressure_w_cop(pressure, cop_mat, com_mat, cop_ipman)
#
#         # import pdb; pdb.set_trace()

