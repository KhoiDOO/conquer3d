import torch
import meshio

def read_obj(file_obj):
    """
    Reads an OBJ file and returns vertices, faces, and optional vertex colors using meshio.
    
    Args:
        file_obj: A file-like object or a string path.
    """
    mesh = meshio.read(file_obj, file_format="obj")
    vertices = torch.tensor(mesh.points, dtype=torch.float32)
    
    faces = None
    if "triangle" in mesh.cells_dict:
        faces = torch.tensor(mesh.cells_dict["triangle"], dtype=torch.long)
    else:
        faces = torch.empty((0, 3), dtype=torch.long)
        
    # Meshio sometimes places vertex colors in point_data under 'obj:vc'
    colors = None
    if mesh.point_data is not None and "obj:vc" in mesh.point_data:
        colors = torch.tensor(mesh.point_data["obj:vc"], dtype=torch.float32)
        
    return vertices, faces, colors

def write_obj(filepath, vertices, faces, colors=None):
    """
    Writes vertices, faces, and optional vertex colors to an OBJ file using meshio.
    """
    point_data = {}
    if colors is not None:
        point_data["obj:vc"] = colors.detach().cpu().numpy()
        
    mesh = meshio.Mesh(
        points=vertices.detach().cpu().numpy(),
        cells=[("triangle", faces.detach().cpu().numpy())],
        point_data=point_data if len(point_data) > 0 else None
    )
    meshio.write(filepath, mesh, file_format="obj")


