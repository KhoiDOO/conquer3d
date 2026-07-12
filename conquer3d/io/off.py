import torch
import trimesh

def read_off(filepath_or_filelike):
    """
    Read an OFF (Object File Format) file and return vertices and faces as PyTorch tensors using trimesh.
    
    Args:
        filepath_or_filelike: A string path or a file-like object (including binary streams).
        
    Returns:
        tuple: (vertices, faces) as torch.Tensor
    """
    # trimesh handles both string paths and binary file streams natively
    mesh = trimesh.load(filepath_or_filelike, file_type='off', process=False, force='mesh')
    
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)
        
    return vertices, faces
