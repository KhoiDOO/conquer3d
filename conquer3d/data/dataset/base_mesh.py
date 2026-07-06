from torch.utils.data import Dataset
from typing import Callable, Optional

class BaseMeshDataset(Dataset):
    """
    Base PyTorch Dataset class for Mesh datasets.
    """
    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__()
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses must implement __len__")

    def __getitem__(self, idx: int):
        raise NotImplementedError("Subclasses must implement __getitem__")
