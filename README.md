# CooPre: Cooperative Pretraining for V2X Cooperative Perception
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2408.11241)

[Seth Z. Zhao](https://sethzhao506.github.io/), [Hao Xiang](https://scholar.google.com/citations?user=04j4RzkAAAAJ&hl=en), [Chenfeng Xu](https://scholar.google.com/citations?user=RpqvaTUAAAAJ&hl=en), [Xin Xia](https://scholar.google.com/citations?user=vCYqMTIAAAAJ&hl=en), [Bolei Zhou](https://boleizhou.github.io/), [Jiaqi Ma](https://mobility-lab.seas.ucla.edu/about/)


This is the official implementation of IROS 2025 paper "CooPre: Cooperative Pretraining for V2X Cooperative Perception". In this paper, we present a multi-agent self-supervised learning framwork for V2X cooperative perception, which utilizes the vast amount of unlabeled 3D V2X data to enhance the perception performance. Our study underscores the critical role of well-learned 3D representations as a promising complement to task-specific design optimizations in V2X cooperative perception.

![teaser](assets/coopre_teaser.png)

## Release Plan
- **`2025/07`**: Full Codebase Release.
- **`2025/04`**: Official Repo Release.

## News
- **`2025/07`**: CooPre has been accepted to IROS 2025 as oral presentation.
- **`2025/06`**: CooPre has been awarded with **Best Paper Award** at the [CVPR 2025 DriveX Workshop](https://drivex-workshop.github.io/).

## Data Download
Please check [website](https://mobility-lab.seas.ucla.edu/v2x-real/) to download the data. The data is in OPV2V format. 

After downloading the data, please put the data in the following structure:
```shell
├── v2xreal
│   ├── train
|      |── 2023-03-17-15-53-02_1_0
│   ├── validate
│   ├── test
```
## Tutorial
#### Environment setup
Please refer to the following steps for the environment setup:
```shell
# Create conda environment (python >= 3.7)
conda create -n coopre python=3.8
conda activate coopre
# pytorch installation
pip3 install torch torchvision torchaudio
# spconv 2.x Installation
pip install spconv-cu120
# Install other dependencies
pip install -r requirements.txt
python setup.py develop
# Install bbx nms calculation cuda version
python opencood/utils/setup.py build_ext --inplace
```

#### Running instructions
For pretraining, please run:
```
bash scripts/pretrain.sh
```

For finetuning, please run:
```
bash scripts/finetune.sh
```

For inference, please run:
```
bash scripts/eval.sh
```

## Acknowledgement
CooPre belongs to the [OpenCDA ecosystem](https://arxiv.org/abs/2301.07325) family. The codebase is built upon [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) in the OpenCDA ecosystem family, and the [V2X-Real](https://arxiv.org/abs/2403.16034), another project in OpenCDA, serves as one of the data sources for this project.

## Citation
If you find this repository useful for your research, please consider giving us a star 🌟 and citing our paper.
 ```bibtex
@article{zhao2024coopre,
  title={CooPre: Cooperative Pretraining for V2X Cooperative Perception},
  author={Zhao, Seth Z and Xiang, Hao and Xu, Chenfeng and Xia, Xin and Zhou, Bolei and Ma, Jiaqi},
  journal={arXiv preprint arXiv:2408.11241},
  year={2024}
}
```
