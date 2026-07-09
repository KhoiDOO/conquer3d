import torch

class BTriangleMesh:
    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor, vertbids: torch.Tensor, facebids: torch.Tensor, batch_size: int):
        self.vertices = vertices
        self.faces = faces
        self.vertbids = vertbids
        self.facebids = facebids
        self.batch_size = batch_size

    def to(self, device):
        self.vertices = self.vertices.to(device)
        self.faces = self.faces.to(device)
        self.vertbids = self.vertbids.to(device)
        self.facebids = self.facebids.to(device)
        return self
        
    def cuda(self, non_blocking=False):
        self.vertices = self.vertices.cuda(non_blocking=non_blocking)
        self.faces = self.faces.cuda(non_blocking=non_blocking)
        self.vertbids = self.vertbids.cuda(non_blocking=non_blocking)
        self.facebids = self.facebids.cuda(non_blocking=non_blocking)
        return self
