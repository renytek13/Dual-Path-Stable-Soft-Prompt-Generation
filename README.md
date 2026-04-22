# Dual-Path Stable Soft Prompt Generation (DPSPG)

Please follow instrcutions below to reproduce the results. 

<hr />

## Installation 
For installation and other package requirements, please follow the instructions as follows. 
This codebase is tested on Ubuntu 20.04 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

* Setup conda environment.
```bash
# Create a conda environment
conda create -y -n spg python=3.8

# Activate the environment
conda activate spg

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/get-started/previous-versions/ if your cuda version is different
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation
# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Clone DPSPG code repository and install requirements.
```bash
# Clone DPSPG code base
git clone https://github.com/renytek13/Dual-Path-Stable-Soft-Prompt-Generation.git
cd Dual-Path-Stable-Soft-Prompt-Generation

# Install requirements
pip install -r requirements.txt
```


## Data Preparation
**Please download the datasets `PACS`, `VLCS`, `Office-Home`, `TerraIncognita` and `DomainNet`.**

You better place all datasets under the same folder `$DATA` for management. We describe how to install the PACS dataset as follows:
- Create a folder named `PACS/` under `$DATA`.
- Download `pacs.zip` from https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE and extract the folder `pacs/images/`. Then put the folder `images/` under `PACS/`.
- Put the given folder `dpspg_coop_splits/` under `$DATA/PACS`.

The organized directory structure is as follows:
```
$DATA/PACS/
|–– images/
|   |–– art_painting/
|   |–– cartoon/
|   |–– photo/
|   |–– sketch/
|–– dpspg_coop_splits/
```


## Training and Evaluation

We provide the running scripts in `scripts`, which allow you to reproduce the results on the paper.

### Dual-Path Stable Domain Prompt Labels Learning

To obtain domain positive and negative prompt labels, please run the bash file in [scripts folder](scripts/dpspg_coop) as follows:
```bash
# Example: trains on PACS dataset with ResNet50 as the backbone. 
bash scripts/dpspg_coop/dpspg_coop.sh pacs RN50
```


### Transformer-based Prompt Generator Pre-training

Please run the bash file in [scripts folder](scripts/dpspg_transformer) as follows.
```bash
# Example: trains on PACS dataset with ResNet50 as the backbone. 
bash scripts/dpspg_transformer/dpspg_transformer.sh pacs RN50
```


### Evaluation
Please run the bash file in [scripts folder](scripts) as follows.
```bash
# Example: test PACS dataset with ResNet50 as the backbone. 
bash scripts/test.sh pacs dpspg_transformer RN50
```
