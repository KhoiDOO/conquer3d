import os
import urllib.request
import torch
from conquer3d.io.obj import read_obj

class Common3D:
    """
    Base class for models from the common-3d-test-models repository.
    Downloads the specific .obj model directly via the Raw GitHub URL
    and loads its vertices, faces, and optional colors.
    """
    def __init__(self, filename, download_dir="~/.conquer3d"):
        self.filename = filename
        self.url = f"https://raw.githubusercontent.com/KhoiDOO/common-3d-test-models/master/data/{self.filename}"
        self.download_dir = os.path.expanduser(download_dir)
        self.vertices = None
        self.faces = None
        self.colors = None
        
        self._load()

    def _load(self):
        os.makedirs(self.download_dir, exist_ok=True)
        obj_path = os.path.join(self.download_dir, self.filename)
        
        if not os.path.exists(obj_path):
            print(f"Downloading {self.filename} from {self.url}...")
            req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(obj_path, 'wb') as out_file:
                out_file.write(response.read())
            print("Download complete.")
            
        print(f"Reading {self.filename}...")
        self.vertices, self.faces, self.colors = read_obj(obj_path)

    def get(self):
        return (
            self.vertices.clone(),
            self.faces.clone(),
            self.colors.clone() if self.colors is not None else None
        )

# --- Child Classes ---

class Alligator(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("alligator.obj", download_dir)

class Armadillo(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("armadillo.obj", download_dir)

class Beast(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("beast.obj", download_dir)

class BeetleAlt(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("beetle-alt.obj", download_dir)

class Beetle(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("beetle.obj", download_dir)

class Bimba(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("bimba.obj", download_dir)

class Cheburashka(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("cheburashka.obj", download_dir)

class Cow(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("cow.obj", download_dir)

class Fandisk(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("fandisk.obj", download_dir)

class HappyBuddha(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("happy.obj", download_dir)

class Homer(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("homer.obj", download_dir)

class Horse(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("horse.obj", download_dir)

class Igea(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("igea.obj", download_dir)

class Lucy(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("lucy.obj", download_dir)

class MaxPlanck(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("max-planck.obj", download_dir)

class Nefertiti(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("nefertiti.obj", download_dir)

class Ogre(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("ogre.obj", download_dir)

class RockerArm(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("rocker-arm.obj", download_dir)

class Spot(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("spot.obj", download_dir)

class StanfordBunny(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("stanford-bunny.obj", download_dir)

class Suzanne(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("suzanne.obj", download_dir)

class Teapot(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("teapot.obj", download_dir)

class Woody(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("woody.obj", download_dir)

class XYZRGBDragon(Common3D):
    def __init__(self, download_dir="~/.conquer3d"):
        super().__init__("xyzrgb_dragon.obj", download_dir)
