import time

import torch
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image as imwrite

import config
import myutils
from loss import Loss
import shutil
import os

from model.VFIformer_arch import VFIformerSmall

model = VFIformerSmall(args)
model = torch.nn.DataParallel(model).to(device)
load_checkpoint(args, model, optimizer, save_loc + '/model_best1.pth')

print(list(model.modules()))
