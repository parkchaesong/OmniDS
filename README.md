# OmniDS: Dual-Stream Context Fusion for Omnidirectional Depth from Fisheye Cameras

Official implementation of our ECCV 2026 paper.

**Project page:** https://parkchaesong.github.io/omnids/

---


## Preparation
#### Installation
Create the environment
```bash
conda create -n omnids python=3.9
conda activate omnids
```
Install pytorch
```bash
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
```
Install other requirements
```bash
pip install -r requirements.txt
```

Then install MultiScaleDeformableAttention
```bash
git clone https://github.com/fundamentalvision/Deformable-DETR.git
cd Deformable-DETR
cd models/ops
sh make.sh
```

#### Download Datasets 
Download the datasets from [dataset link](https://rvlab.snu.ac.kr/research/omnistereo)


## Evaluation  
Download the checkpoint files from [Google Drive](https://drive.google.com/drive/folders/1L_-uJQsH0_bKX_3gHYsyBAYAIxxc6K3d?usp=sharing) and place them in `checkpoints/`.

Test with non-distilled model *(trained on OmniThings dataset only)*
```bash
python test.py --ckpt checkpoints/omnids.pth --dbname {DATASET_NAME} --mixed_precision
```

Test with non-distilled model *(fine-tuned on OmniHouse and Sunny datasets)*
```bash
python test.py --ckpt checkpoints/omnids_ft.pth --dbname {DATASET_NAME} --mixed_precision
```

Test with distilled model 
```bash
python test.py --ckpt checkpoints/distilled_e14.pth --dbname {DATASET_NAME} --mixed_precision --distilled
```

## Acknowledgements
The project borrows codes from [OmniMVS](https://github.com/hyu-cvlab/omnimvs-pytorch) and [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo). Many thanks to their authors. 