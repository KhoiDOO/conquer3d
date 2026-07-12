import torch
import meshio

def read_off(filepath_or_filelike):
    """
    Read an OFF (Object File Format) file and return vertices and faces as PyTorch tensors using meshio.
    
    Args:
        filepath_or_filelike: A string path or a file-like object.
        
    Returns:
        tuple: (vertices, faces) as torch.Tensor
    """
    mesh = meshio.read(filepath_or_filelike, file_format="off")
    vertices = torch.tensor(mesh.points, dtype=torch.float32)
    
    faces = None
    if "triangle" in mesh.cells_dict:
        faces = torch.tensor(mesh.cells_dict["triangle"], dtype=torch.long)
    else:
        faces = torch.empty((0, 3), dtype=torch.long)
        
    return vertices, faces
