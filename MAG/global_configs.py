import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROGRAM"] = "multimodal_driver.py"
# os.environ["WANDB_MODE"] = dryrun
# WANDB_MODE="dryrun"
#
DEVICE = torch.device("cuda:1")

# MOSI SETTING
# ACOUSTIC_DIM = 74
# VISUAL_DIM = 47
# TEXT_DIM = 768

# MOSEI SETTING
ACOUSTIC_DIM = 74
VISUAL_DIM = 35
TEXT_DIM = 768

XLNET_INJECTION_INDEX = 1

DATA_DICT = '/home/iknow/workspace/multimodal'