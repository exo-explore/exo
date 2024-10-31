sudo apt-get update -y
sudo ubuntu-drivers autoinstall
sudo apt install nvidia-cuda-toolkit -y

sudo apt install python3.10-venv -y
sudo apt-get -y install --no-install-recommends clang python3-clang
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.debsudo dpkg -i cuda-keyring_1.1-1_all.debsudo apt-get updatesudo apt-get -y install cudnn
git clone https://github.com/RashikShahjahan/exo.git
cd exo && git checkout quantized-models-support-tinygrad
source install.sh
pip install torch llvmlite

