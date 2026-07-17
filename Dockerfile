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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Create the conda environment with compilers
RUN conda create -c conda-forge -n geocutool python=3.10 gxx_linux-64=13 gcc_linux-64=13 -y

# Make RUN commands use the new conda environment
SHELL ["conda", "run", "-n", "geocutool", "/bin/bash", "-c"]

# Install conda dependencies: sparsehash and cuda-toolkit
RUN conda install -c conda-forge sparsehash -y && \
    conda install nvidia::cuda-toolkit==12.8.2 -y

# Install PyTorch and build tools
RUN pip install setuptools wheel ninja && \
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install heavy 3D and graphics dependencies
RUN pip install git+https://github.com/mit-han-lab/torchsparse.git --no-build-isolation && \
    pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html && \
    pip install git+https://github.com/NVlabs/nvdiffrast.git

# Install visualization and meshing utilities
RUN pip install pybind11-stubgen plotly open3d jupyter trimesh point-cloud-utils meshlib pymeshlab kiui rectified-flow-pytorch

# Copy only the necessary files for compiling
COPY setup.py pyproject.toml README.md /workspace/
COPY conquer3d /workspace/conquer3d/

# Install geocutool
WORKDIR /workspace
RUN pip install -e . --no-build-isolation

# Set default command to bash with conda activated
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "geocutool"]
CMD ["/bin/bash"]
