"""
PyRender-based Visualization utilities for SMPL meshes with pressure data.

This module provides tools to visualize SMPL body meshes from vertices or SMPL parameters,
with optional checkerboard floor for better spatial reference, using pyrender for better
rendering quality.
"""

import torch
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict, Tuple
import smplx
import os

# Handle headless environments
try:
    # For headless rendering on servers
    if 'DISPLAY' not in os.environ:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
except:
    pass

from lib.utils.static import SMPL_MODEL


def get_checkerboard_plane(plane_width=5, num_boxes=10, center=True):
    pw = plane_width / num_boxes
    white = [220, 220, 220, 255]
    # black = [180, 210, 230, 255]
    black = [25, 25, 25, 255]
    
    meshes = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            # 计算当前格子的中心坐标
            cx = i * pw + (pw / 2)
            cy = j * pw + (pw / 2)
            
            if center:
                cx -= (plane_width / 2)
                cy -= (plane_width / 2)
            
            # 创建一个极薄的盒子作为格子
            # 使用 Box 保证了法线和面是独立的
            ground = trimesh.primitives.Box(
                extents=[pw, pw, 0.0001]
            )
            
            # 移动到正确的位置 (XY平面)
            ground.apply_translation([cx, cy, 0])
            
            # 直接给面赋色（这是关键，避免了顶点插值）
            ground.visual.face_colors = black if ((i + j) % 2) == 0 else white
            meshes.append(ground)
    
    # 重点：将所有小盒子合并成一个单一的 trimesh 对象
    # 这样在 pyrender 中只需创建一个节点，渲染效率更高
    combined_trimesh = trimesh.util.concatenate(meshes)
    
    # 返回 pyrender.Mesh 对象
    return pyrender.Mesh.from_trimesh(combined_trimesh, smooth=False)


class PyRenderVisualizer:
    """PyRender-based visualizer for SMPL meshes with optional checkerboard floor."""
    
    def __init__(
        self,
        smpl_model_path: str = SMPL_MODEL,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize the mesh visualizer.
        
        Args:
            smpl_model_path: Path to SMPL model files
            device: Device for computations
        """
        self.device = device
        self.smpl_model_path = smpl_model_path
        
        # Initialize SMPL models
        self.male_model = smplx.create(
            smpl_model_path,
            model_type='smpl',
            gender='male',
            ext='pkl'
        ).to(device)
        
        self.female_model = smplx.create(
            smpl_model_path,
            model_type='smpl',
            gender='female',
            ext='pkl'
        ).to(device)

        self.neutral_model = smplx.create(
            smpl_model_path,
            model_type='smpl',
            gender='neutral',
            ext='pkl'
        ).to(device)
        
        # Try to get SMPL faces (standard SMPL has 13776 faces)
        try:
            self.smpl_faces = self.male_model.faces
        except:
            # Fallback if faces are not directly accessible
            # Standard SMPL face indices
            self.smpl_faces = np.loadtxt(os.path.join(smpl_model_path, 'smpl_faces.txt')).astype(np.int32)
    
    def vertices_to_mesh(
        self,
        vertices: Union[np.ndarray, torch.Tensor],
        faces: Optional[np.ndarray] = None
    ) -> pyrender.Mesh:
        """
        Convert vertices to a pyrender mesh object.
        
        Args:
            vertices: Vertices array of shape (N, 3) or (3, N)
            faces: Face indices array of shape (F, 3). If None, uses SMPL faces.
            
        Returns:
            Pyrender mesh object
        """
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
            
        # Ensure vertices are in the right shape (N, 3)
        if vertices.shape[0] == 3 and vertices.shape[1] != 3:
            vertices = vertices.T
            
        # Use provided faces or default SMPL faces
        if faces is None:
            faces = self.smpl_faces
            
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Convert to pyrender mesh
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        
        return pyrender_mesh
    
    def smpl_to_vertices(
        self,
        global_orient: Union[np.ndarray, torch.Tensor],
        body_pose: Union[np.ndarray, torch.Tensor],
        betas: Union[np.ndarray, torch.Tensor],
        transl: Union[np.ndarray, torch.Tensor],
        gender: str = 'male'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert SMPL parameters to vertices and faces.
        
        Args:
            global_orient: Global orientation (3,)
            body_pose: Body pose parameters (69,)
            betas: Shape parameters (10,)
            transl: Translation (3,)
            gender: Gender ('male' or 'female')
            
        Returns:
            Tuple of (vertices, faces)
        """
        # Convert to tensors if needed
        if isinstance(global_orient, np.ndarray):
            global_orient = torch.from_numpy(global_orient).float()
        if isinstance(body_pose, np.ndarray):
            body_pose = torch.from_numpy(body_pose).float()
        if isinstance(betas, np.ndarray):
            betas = torch.from_numpy(betas).float()
        if isinstance(transl, np.ndarray):
            transl = torch.from_numpy(transl).float()
            
        # Add batch dimension if needed
        if global_orient.dim() == 1:
            global_orient = global_orient.unsqueeze(0)
        if body_pose.dim() == 1:
            body_pose = body_pose.unsqueeze(0)
        if betas.dim() == 1:
            betas = betas.unsqueeze(0)
        if transl.dim() == 1:
            transl = transl.unsqueeze(0)

        # Move to device
        global_orient = global_orient.to(self.device)
        body_pose = body_pose.to(self.device)
        betas = betas.to(self.device)
        transl = transl.to(self.device)
        
        # Select model based on gender
        if gender.lower() == 'female':
            model = self.female_model
        elif gender.lower() == 'male':
            model = self.male_model
        else:
            model = self.neutral_model
        
        # Generate SMPL output
        with torch.no_grad():
            output = model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl
            )
            
        # Extract vertices
        vertices = output.vertices.squeeze().cpu().numpy()
        faces = self.smpl_faces
        
        return vertices, faces
    
    def visualize_mesh(
        self,
        vertices: Optional[Union[np.ndarray, torch.Tensor]] = None,
        faces: Optional[np.ndarray] = None,
        global_orient: Optional[Union[np.ndarray, torch.Tensor]] = None,
        body_pose: Optional[Union[np.ndarray, torch.Tensor]] = None,
        betas: Optional[Union[np.ndarray, torch.Tensor]] = None,
        transl: Optional[Union[np.ndarray, torch.Tensor]] = None,
        gender: str = 'male',
        show_floor: bool = True,
        floor_size: float = 10.0,
        floor_subdivisions: int = 6,
        **kwargs
    ) -> bool:
        # 1. 顶点处理 (保持你原有的逻辑)
        if vertices is not None:
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.detach().cpu().numpy()
            # Ensure vertices are in the right shape (N, 3)
            if vertices.shape[0] == 3 and vertices.shape[1] != 3:
                vertices = vertices.T
        elif all(param is not None for param in [global_orient, body_pose, betas, transl]):
            vertices, faces = self.smpl_to_vertices(
                global_orient, body_pose, betas, transl, gender
            )
        else:
            raise ValueError("Either vertices or all SMPL parameters must be provided")
        if faces is None:
            faces = self.smpl_faces

        # 2. 创建场景并设置环境光 (Ambient Light 让阴影处不全黑)
        scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[1.0, 1.0, 1.0])

        # 3. 添加 SMPL 人体 (使用带平滑的材质)
        body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        body_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.6,
            baseColorFactor=[0.345, 0.580, 0.713, 1.0]
        )
        body_pr = pyrender.Mesh.from_trimesh(body_mesh, material=body_material)
        scene.add(body_pr)

        # 4. 添加蓝白棋盘格地板 (使用之前优化的 Box 拼接法)
        if show_floor:
            # 这里的 get_checkerboard_plane 内部应返回带哑光材质的 Mesh
            floor_mesh = get_checkerboard_plane(
                plane_width=floor_size, 
                num_boxes=floor_subdivisions
            )
            scene.add(floor_mesh)

        # 5. 设置光照 (单向主光源 + 弱光，关闭 raymond)
        # 放置在人体斜上方
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [2.0, -2.0, 5.0] # 斜后上方俯冲
        scene.add(light, pose=light_pose)

        # 6. 动态计算相机位姿 (核心优化)
        mesh_center = np.mean(vertices, axis=0)
        
        # 定义相机位置 (斜上方 45 度左右)
        # 基于 z 轴向上的坐标系：X(右), Y(前), Z(上)
        camera_pos = mesh_center + np.array([3.0, -4.0, 2.5]) 
        
        # 计算 Look-At 矩阵
        def look_at(eye, target, up):
            zaxis = (eye - target) / np.linalg.norm(eye - target)
            xaxis = np.cross(up, zaxis)
            xaxis /= np.linalg.norm(xaxis)
            yaxis = np.cross(zaxis, xaxis)
            view_mat = np.eye(4)
            view_mat[:3, 0] = xaxis
            view_mat[:3, 1] = yaxis
            view_mat[:3, 2] = zaxis
            view_mat[:3, 3] = eye
            return view_mat

        camera_pose = look_at(camera_pos, mesh_center, up=[0, 0, 1])
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.33)
        scene.add(camera, pose=camera_pose)

        # 7. 渲染显示 (禁用 use_raymond_lighting 以免颜色混乱)
        pyrender.Viewer(scene, use_raymond_lighting=False, viewport_size=(1280, 960))

        return True
    
    def visualize_dataset_sample(
        self,
        sample: Dict[str, torch.Tensor],
        show_floor: bool = True,
        floor_size: float = 10.0,
        floor_subdivisions: int = 10
    ) -> bool:
        """
        Visualize a dataset sample.
        
        Args:
            sample: Dataset sample dictionary with 'vertices' or 'smpl' keys
            show_floor: Whether to show checkerboard floor
            floor_size: Size of the floor
            floor_subdivisions: Number of subdivisions for checkerboard
            
        Returns:
            True if successful, False otherwise
        """
        if 'vertices' in sample:
            # Directly use vertices
            vertices = sample['vertices']
            if vertices.dim() > 2:
                vertices = vertices[0]  # Take first in batch
                
            self.visualize_mesh(
                vertices=vertices,
                show_floor=show_floor,
                floor_size=floor_size,
                floor_subdivisions=floor_subdivisions,
                title="Dataset Sample - Vertices"
            )
        if 'smpl' in sample:
            # Extract SMPL parameters
            smpl_params = sample['smpl']
            if smpl_params.dim() > 1:
                smpl_params = smpl_params[0]  # Take first in batch
                
            # Parse SMPL parameters (assuming standard SMPL format)
            global_orient = smpl_params[:3]
            body_pose = smpl_params[3:72]
            betas = smpl_params[72:82]
            transl = smpl_params[82:85]
            
            # import pdb; pdb.set_trace()

            # Get gender if available
            gender = 'neutral'
            if 'gender' in sample:
                gender_val = sample['gender']
                if gender_val.dim() > 0:
                    gender_val = gender_val[0]
                gender = 'female' if gender_val == 0 else 'male'
                
            self.visualize_mesh(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl,
                gender=gender,
                show_floor=show_floor,
                floor_size=floor_size,
                floor_subdivisions=floor_subdivisions,
                title="Dataset Sample - SMPL Parameters"
            )
        
        return True

# Convenience function for quick visualization
def visualize_mesh_from_vertices(
    vertices: Union[np.ndarray, torch.Tensor],
    show_floor: bool = True,
    floor_size: float = 10.0,
    floor_subdivisions: int = 10
) -> bool:
    """
    Quick visualization of mesh from vertices.
    
    Args:
        vertices: Vertices array
        show_floor: Whether to show checkerboard floor
        floor_size: Size of the floor
        floor_subdivisions: Number of subdivisions for checkerboard
        
    Returns:
        True if successful, False otherwise
    """
    visualizer = PyRenderVisualizer()
    return visualizer.visualize_mesh(
        vertices=vertices,
        show_floor=show_floor,
        floor_size=floor_size,
        floor_subdivisions=floor_subdivisions
    )


def visualize_mesh_from_smpl(
    global_orient: Union[np.ndarray, torch.Tensor],
    body_pose: Union[np.ndarray, torch.Tensor],
    betas: Union[np.ndarray, torch.Tensor],
    transl: Union[np.ndarray, torch.Tensor],
    gender: str = 'male',
    show_floor: bool = True,
    floor_size: float = 10.0,
    floor_subdivisions: int = 10
) -> bool:
    """
    Quick visualization of mesh from SMPL parameters.
    
    Args:
        global_orient: Global orientation (3,)
        body_pose: Body pose parameters (69,)
        betas: Shape parameters (10,)
        transl: Translation (3,)
        gender: Gender ('male' or 'female')
        show_floor: Whether to show checkerboard floor
        floor_size: Size of the floor
        floor_subdivisions: Number of subdivisions for checkerboard
        
    Returns:
        True if successful, False otherwise
    """
    visualizer = PyRenderVisualizer()
    return visualizer.visualize_mesh(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        transl=transl,
        gender=gender,
        show_floor=show_floor,
        floor_size=floor_size,
        floor_subdivisions=floor_subdivisions
    )