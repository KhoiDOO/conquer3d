import os
import urllib.request
import zipfile
import torch
from conquer3d.io.off import read_off

class Iphiagenia:
    """
    Iphigenia asset mesh from pmp-book.org.
    Downloads the full resolution Iphigenia mesh (zip), extracts the .off file,
    and loads the vertices and faces into PyTorch tensors.
    """
    def __init__(self, download_dir="~/.conquer3d"):
        self.url = "https://www.pmp-book.org/download/meshes/iphi_fullres.zip"
        self.download_dir = os.path.expanduser(download_dir)
        self.vertices = None
        self.faces = None
        
        self._load()

    def _load(self):
        os.makedirs(self.download_dir, exist_ok=True)
        zip_path = os.path.join(self.download_dir, "iphi_fullres.zip")
        
        if not os.path.exists(zip_path):
            print(f"Downloading Iphigenia mesh from {self.url}...")
            # We use a Request with a User-Agent to easily bypass basic checks
            req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
            print("Download complete.")
            
        print("Extracting and reading OFF file...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Find the .off file in the zip
            off_filename = next((name for name in z.namelist() if name.lower().endswith('.off')), None)
            if not off_filename:
                raise FileNotFoundError("Could not find an .off file in the downloaded zip.")
                
            with z.open(off_filename, 'r') as off_file:
                # off_file is a file-like object of bytes.
                # read_off will automatically decode and parse it.
                self.vertices, self.faces = read_off(off_file)
