FROM ghcr.io/actions/actions-runner:latest

ENV CXX=clang++

RUN sudo apt-get update -y \
    && sudo apt-get install -y software-properties-common \
    && sudo add-apt-repository -y ppa:git-core/ppa \
    && sudo apt-get update -y \
    && sudo apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    jq \
    sudo \
    unzip \
    zip \
    cmake \
    ninja-build \
    clang \
    lld \
    wget \
    psmisc \
    python3.10-venv \
    && sudo rm -rf /var/lib/apt/lists/*

RUN sudo apt-get update && sudo apt-get install -y python3.10 python3-pip python-is-python3 python3-setuptools python3-wheel libpython3.10

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
    sudo apt-get install git-lfs

RUN sudo groupadd -g 109 render

RUN sudo apt update -y \
    && sudo apt install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)" \
    && sudo usermod -a -G render,video runner \
    && wget https://repo.radeon.com/amdgpu-install/6.3.1/ubuntu/jammy/amdgpu-install_6.3.60301-1_all.deb \
    && sudo apt install -y ./amdgpu-install_6.3.60301-1_all.deb \
    && sudo apt update -y \
    && sudo apt install -y rocm

RUN sudo pip install --upgrade pip

RUN sudo pip install --no-cache-dir pytorch-triton-rocm torch --index-url https://download.pytorch.org/whl/nightly/rocm6.3

RUN git clone --recursive https://github.com/ROCm/aiter.git \
    && cd aiter \
    && git checkout 1d88633958236e942cba3c283864282f7af3ebc5 \
    && sudo pip install -r requirements.txt \
    && sudo python3 setup.py develop

RUN sudo pip install \
    ninja \
    numpy \
    packaging \
    wheel \
    tinygrad
