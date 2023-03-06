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
from dataset.atd12k import get_loader


args, unparsed = config.get_args()
device = torch.device('cuda' if args.cuda else 'cpu')
train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)
test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)


model = VFIformerSmall(args)
model = torch.nn.DataParallel(model).to(device)
load_checkpoint(args, model, optimizer, save_loc + '/model_best1.pth')

print(list(model.modules()))
