# install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 755 ./Miniconda3-latest-Linux-x86_64.sh 
bash ./Miniconda3-latest-Linux-x86_64.sh 

# reload bash
. ~/.bashrc

# create conda environment (sandbox)
conda update -n base -c defaults conda
conda create -n sandbox python=3
conda install pytorch torchvision cudatoolkit jupyterlab numpy matplotlib torchaudio -c pytorch -y
conda activate sandbox

# check if GPUs accisble to pytorch
python -c 'import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print("cuda is not available")'