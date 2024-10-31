sudo apt-get update -y
sudo ubuntu-drivers autoinstall
sudo apt install nvidia-cuda-toolkit -y
sudo apt install python3-pip -y
sudo apt install python3.10-venv -y
sudo apt-get -y install --no-install-recommends clang python3-clang
git clone https://github.com/RashikShahjahan/exo.git
cd exo && git checkout quantized-models-support-tinygrad
source install.sh
pip install torch llvmlite

export SUPPORT_BF16=0
python3 test/test_inference.py
