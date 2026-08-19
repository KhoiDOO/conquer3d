"""Standard 3D benchmark model assets loader.

This module provides automatic download and loading interfaces for standard
geometry processing benchmark models (Stanford Bunny, Dragon, Armadillo, Cow, Spot, etc.).
"""

from typing import Tuple, Optional
import os
import urllib.request
import torch
from conquer3d.io.obj import read_obj


class Common3D:
    """Base class for benchmark test models from the common-3d-test-models repository.

    Downloads the specific `.obj` model directly via the raw GitHub URL
    and loads its vertices, faces, and optional vertex colors.

    Attributes:
        filename (str): Name of the remote `.obj` file.
        download_dir (str): Local cache directory.
        vertices (torch.Tensor | None): Float32 tensor of shape `(V, 3)`.
        faces (torch.Tensor | None): Int64 tensor of shape `(F, 3)`.
        colors (torch.Tensor | None): Optional float32 tensor of shape `(V, 3)`.
    """

    def __init__(self, filename: str, download_dir: str = "~/.conquer3d", verbose: bool = False) -> None:
        """Initializes and automatically downloads/caches the model.

        Args:
            filename (str): Name of the `.obj` asset.
            download_dir (str, optional): Target local directory. Defaults to `"~/.conquer3d"`.
            verbose (bool, optional): Whether to print download progress. Defaults to False.
        """
        self.filename = filename
        self.url = f"https://raw.githubusercontent.com/KhoiDOO/common-3d-test-models/master/data/{self.filename}"
        self.download_dir = os.path.expanduser(download_dir)
        self.vertices = None
        self.faces = None
        self.colors = None
        self.verbose = verbose

        self._load()

    def _load(self) -> None:
        os.makedirs(self.download_dir, exist_ok=True)
        obj_path = os.path.join(self.download_dir, self.filename)
        
        if not os.path.exists(obj_path):
            if self.verbose:
                print(f"Downloading {self.filename} from {self.url}...")
            req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(obj_path, 'wb') as out_file:
                out_file.write(response.read())
            if self.verbose:
                print("Download complete.")
            
        if self.verbose:
            print(f"Reading {self.filename}...")
        self.vertices, self.faces, self.colors = read_obj(obj_path)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Retrieves clones of the loaded geometry tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - vertices: (V, 3) float32 coordinates.
                - faces: (F, 3) int64 face indices.
                - colors: (V, 3) float32 colors or None.
        """
        return (
            self.vertices.clone(),
            self.faces.clone(),
            self.colors.clone() if self.colors is not None else None
        )


class Alligator(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("alligator.obj", download_dir)


class Armadillo(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("armadillo.obj", download_dir)


class Beast(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("beast.obj", download_dir)


class Beetle(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("beetle.obj", download_dir)


class Bimba(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("bimba.obj", download_dir)


class Bob(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("bob.obj", download_dir)


class Bunny(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("bunny.obj", download_dir)


class Camel(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("camel.obj", download_dir)


class Cow(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("cow.obj", download_dir)


class Dinosaur(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("dinosaur.obj", download_dir)


class Dragon(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("dragon.obj", download_dir)


class Duck(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("duck.obj", download_dir)


class Feline(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("feline.obj", download_dir)


class Frog(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("frog.obj", download_dir)


class Horse(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("horse.obj", download_dir)


class Lucy(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("lucy.obj", download_dir)


class Nefertiti(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("nefertiti.obj", download_dir)


class Spot(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("spot.obj", download_dir)


class StanfordBunny(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("stanford-bunny.obj", download_dir)


class StanfordDragon(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("stanford-dragon.obj", download_dir)


class Teapot(Common3D):
    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        super().__init__("teapot.obj", download_dir)
