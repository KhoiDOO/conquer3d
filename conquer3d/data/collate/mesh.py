import torch
from conquer3d.data_structure.bmesh import BTriangleMesh

def bmesh_collate_fn(batch):
    """
    Collate function for a dataset returning (vertices, faces, label).
    Returns a BTriangleMesh and a batched label tensor.
    """
    all_vertices = []
    all_faces = []
    all_vertbids = []
    all_facebids = []
    all_labels = []
    
    for b, (v, f, l) in enumerate(batch):
        all_vertices.append(v)
        
        # We do not offset the face indices because we will process them per-mesh locally
        all_faces.append(f)
        
        all_vertbids.append(torch.full((v.shape[0],), b, dtype=torch.int32))
        all_facebids.append(torch.full((f.shape[0],), b, dtype=torch.int32))
        all_labels.append(l)
        
    batched_vertices = torch.cat(all_vertices, dim=0)
    batched_faces = torch.cat(all_faces, dim=0)
    batched_vertbids = torch.cat(all_vertbids, dim=0)
    batched_facebids = torch.cat(all_facebids, dim=0)
    
    batched_labels = torch.tensor(all_labels, dtype=torch.long)
    
    bmesh = BTriangleMesh(
        vertices=batched_vertices,
        faces=batched_faces,
        vertbids=batched_vertbids,
        facebids=batched_facebids,
        batch_size=len(batch)
    )
    
    return bmesh, batched_labels
