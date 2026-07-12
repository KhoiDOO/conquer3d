import torch
import trimesh
import numpy as np

def read_obj(file_obj):
    """
    Reads an OBJ file and returns vertices, faces, and optional vertex colors using trimesh.
    
    Args:
        file_obj: A file-like object or a string path.
    """
    # trimesh safely handles UV seams and mismatched v/vt counts
    mesh = trimesh.load(file_obj, process=False, force='mesh', skip_materials=True)
    
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)
    
    colors = None
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
        colors = torch.tensor(mesh.visual.vertex_colors[:, :3], dtype=torch.float32) / 255.0
        
    return vertices, faces, colors

def write_obj(filepath, vertices, faces, colors=None):
    """
    Writes vertices, faces, and optional vertex colors to an OBJ file using trimesh.
    """
    v = vertices.detach().cpu().numpy()
    f = faces.detach().cpu().numpy()
    
    vc = None
    if colors is not None:
        vc = (colors.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        
    mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=vc, process=False)
    mesh.export(filepath)


