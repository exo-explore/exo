sudo apt-get update -y
sudo apt install nvidia-cuda-toolkit -y
sudo apt install python3-pip -y
sudo apt install python3-venv -y
sudo apt-get -y install --no-install-recommends clang python3-clang
reboot
wget https://developer.download.nvidia.com/compute/cudnn/9.5.1/local_installers/cudnn-local-repo-ubuntu2204-9.5.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.5.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.5.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update -y
sudo apt-get -y install cudnn
reboot
sudo ubuntu-drivers autoinstall
reboot
git clone https://github.com/RashikShahjahan/exo.git
cd exo && git checkout quantized-models-support-tinygrad
source install.sh
pip install torch llvmlite

python test_quantization.py --model llama-3.1-8b --prompt "What is the meaning of exo?" --quant int8
