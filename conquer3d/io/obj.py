import torch

def read_obj(file_obj):
    """
    Reads an OBJ file and returns vertices, faces, and optional vertex colors.
    
    Args:
        file_obj: A file-like object or a string path.
    """
    vertices = []
    faces = []
    colors = []
    
    close_file = False
    if isinstance(file_obj, str):
        file_obj = open(file_obj, 'r', encoding='utf-8')
        close_file = True

    try:
        for line in file_obj:
            if isinstance(line, bytes):
                line = line.decode('utf-8', errors='ignore')
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
            elif line.startswith('f '):
                parts = line.split()
                # Face elements can be v, v/vt, or v/vt/vn. We just want the first index (v).
                face = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                faces.append(face)
    finally:
        if close_file:
            file_obj.close()
            
    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.long)
    
    if len(colors) == len(vertices):
        colors = torch.tensor(colors, dtype=torch.float32)
    else:
        colors = None
        
    return vertices, faces, colors
