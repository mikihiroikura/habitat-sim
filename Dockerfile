# Base image
FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Set timezone
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Setup basic packages
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp@g' /etc/apt/sources.list && \
    sed -i 's@security.ubuntu.com@ftp.jaist.ac.jp@g' /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository universe \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    unzip \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libglx-mesa0 \
    libxrandr2 \
    libxi6 \
    libxinerama1 \
    libxcursor1 \
    xauth \
    mesa-utils \
    libglvnd-dev \
    xdg-utils && \
    rm -rf /var/lib/apt/lists/*

# Install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get update && apt-get install git-lfs && \
    git lfs install

# Add User ID and Group ID
ARG UNAME=habitat
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

# Add User into sudoers, can run sudo command without password
RUN apt update && apt install -y sudo
RUN usermod -aG sudo ${UNAME}
RUN echo "${UNAME} ALL=(ALL) NOPASSWD:ALL" | tee /etc/sudoers.d/${UNAME}

ARG CODE_DIR=/home/${UNAME}


### Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p $CONDA_DIR && \
    rm /miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Conda environment
RUN conda create -n habitat python=3.9 cmake=3.14.0

# Setup habitat-sim
RUN cd $CODE_DIR && git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
RUN cd $CODE_DIR && /bin/bash -c ". activate habitat; conda install habitat-sim withbullet headless -c conda-forge -c aihabitat"

# Install challenge specific habitat-lab
RUN cd $CODE_DIR && git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
RUN cd $CODE_DIR && /bin/bash -c ". activate habitat; pip install -e habitat-lab/"

### Change owner
RUN chown -R $UNAME:$UNAME ${CODE_DIR} ${CONDA_DIR}

# Install requirements.txt
COPY . $CODE_DIR/habitat-sim
RUN cd $CODE_DIR/habitat-sim && /bin/bash -c ". activate habitat; conda install -n habitat -c conda-forge --file requirements.txt"

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"

### Switch user
USER $UNAME
WORKDIR ${CODE_DIR}

# Initialize conda for bash
RUN conda init bash
RUN echo "conda activate habitat" >> ~/.bashrc