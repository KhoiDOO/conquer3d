# Use Miniconda3 as the base image
FROM continuumio/miniconda3:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1
ENV MAX_JOBS=4

# Install basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    pkg-config \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Create the conda environment with compilers and pip
RUN conda create -c conda-forge -n conquer3d python=3.10 pip gxx_linux-64=13 gcc_linux-64=13 -y
RUN conda install -n conquer3d -c conda-forge sparsehash -y
RUN conda install -n conquer3d nvidia::cuda-toolkit==12.8.2 -y
ENV PATH=/opt/conda/envs/conquer3d/bin:$PATH
SHELL ["conda", "run", "-n", "conquer3d", "/bin/bash", "-c"]

# Install PyTorch and build tools
RUN pip install setuptools wheel ninja && \
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install heavy 3D and graphics dependencies
RUN pip install git+https://github.com/mit-han-lab/torchsparse.git --no-build-isolation
RUN pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
RUN pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

# Install visualization and meshing utilities
RUN pip install pybind11-stubgen plotly open3d jupyter trimesh point-cloud-utils pymeshlab kiui rectified-flow-pytorch

# Others
RUN pip install pytorch-fid

# Copy only the necessary files for compiling
COPY setup.py pyproject.toml README.md /workspace/
COPY conquer3d /workspace/conquer3d/

# Install conquer3d
WORKDIR /workspace
RUN pip install -e . --no-build-isolation

# Set default command to bash with conda activated
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "conquer3d"]
CMD ["/bin/bash"]
