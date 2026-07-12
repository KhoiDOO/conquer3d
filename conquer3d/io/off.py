import torch
import numpy as np
import io

def read_off(filepath_or_filelike):
    """
    Read an OFF (Object File Format) file and return vertices and faces as PyTorch tensors.
    Supports reading from a file path or a file-like object.
    
    Args:
        filepath_or_filelike: A string path or a file-like object.
        
    Returns:
        tuple: (vertices, faces) as torch.Tensor
    """
    if isinstance(filepath_or_filelike, str):
        with open(filepath_or_filelike, "r") as f:
            lines = f.readlines()
    else:
        # Assuming text-based or bytes file-like object
        lines = filepath_or_filelike.readlines()
        if len(lines) > 0 and isinstance(lines[0], bytes):
            lines = [line.decode('utf-8') for line in lines]

    # Clean up empty lines and comments
    lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    if not lines:
        raise ValueError("Empty OFF file.")

    first_line = lines[0].strip()
    if first_line.upper().startswith("OFF"):
        if len(first_line) > 3:
            # Format: 'OFF 123 456 0'
            parts = first_line[3:].strip().split()
            # If the rest of the line doesn't contain counts, maybe they are on the next line
            if parts:
                lines = [parts] + lines[1:]
            else:
                lines = lines[1:]
        else:
            lines = lines[1:]
            
    # If the format parsing above resulted in a list of parts at index 0, handle it
    if isinstance(lines[0], list):
        counts = lines[0]
    else:
        counts = lines[0].split()
        
    num_verts, num_faces, num_edges = map(int, counts[:3])
    
    # Fast parsing with numpy
    verts_lines = lines[1:num_verts + 1]
    verts_str = '\n'.join(verts_lines)
    verts = np.fromstring(verts_str, sep=' ').reshape(num_verts, -1)[:, :3]
    
    faces_lines = lines[num_verts + 1 : num_verts + 1 + num_faces]
    faces_str = '\n'.join(faces_lines)
    
    # In an OFF file, face lines look like: `3 v1 v2 v3` for triangles.
    # We parse the flat array and reshape based on the number of elements per line
    faces_raw = np.fromstring(faces_str, sep=' ', dtype=int)
    
    # We assume all faces are triangles! 
    # The first element of each chunk will be 3, followed by 3 indices.
    # Total length should be num_faces * 4
    if len(faces_raw) == num_faces * 4:
        faces = faces_raw.reshape(num_faces, 4)[:, 1:4]
    else:
        # If faces are mixed (e.g. some quads), we fall back to a slower parsing
        faces_list = []
        idx = 0
        for _ in range(num_faces):
            n_v = faces_raw[idx]
            if n_v == 3:
                faces_list.append([faces_raw[idx+1], faces_raw[idx+2], faces_raw[idx+3]])
            else:
                # Naive triangulation for ngons (fan triangulation)
                for i in range(1, n_v - 1):
                    faces_list.append([faces_raw[idx+1], faces_raw[idx+i+1], faces_raw[idx+i+2]])
            idx += n_v + 1
        faces = np.array(faces_list, dtype=int)
        
    return torch.from_numpy(verts).float(), torch.from_numpy(faces).long()
