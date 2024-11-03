sudo apt-get update -y
sudo apt install nvidia-cuda-toolkit -y
sudo apt install python3-pip

sudo apt install python3-venv -y

sudo apt-get -y install --no-install-recommends clang python3-clang
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.5.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update -y
sudo apt-get -y install cudnn
sudo ubuntu-drivers autoinstall

git clone https://github.com/RashikShahjahan/exo.git
cd exo && git checkout quantized-models-support-tinygrad
source install.sh
pip install torch llvmlite

