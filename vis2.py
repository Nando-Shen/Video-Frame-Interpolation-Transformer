import time

import torch
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image as imwrite
from torchvision import transforms
from PIL import Image

import config
import myutils
from loss import Loss
import shutil
import os

def load_checkpoint(args, model, optimizer, path):
    print("loading checkpoint %s" % path)
    checkpoint = torch.load(path)
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = checkpoint.get("lr", args.lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()
print(args)

save_loc = os.path.join(args.checkpoint_dir, "checkpoints")

device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

from model.VFIformer_arch import VFIformerSmall

print("Building model: %s"%args.model)
args.device = device
args.resume_flownet = False
model = VFIformerSmall(args)

model = torch.nn.DataParallel(model).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('the number of network parameters: {}'.format(total_params))

from torch.optim import Adamax
optimizer = Adamax(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

transforms = transforms.Compose([
                transforms.ToTensor()
            ])

input_dir = '/home/jiaming/sim_keras_init'
result_dir = '/home/jiaming/csi_vis_result'

def read(idx):
    data_list = []
    img0 = os.path.join(input_dir, '{}.jpg'.format(idx))
    img1 = os.path.join(input_dir, '{}.jpg'.format(idx+2))

    points14 = os.path.join(input_dir, '{}-{}points14.jpg'.format(idx,idx+2))
    points12 = os.path.join(input_dir, '{}-{}points12.jpg'.format(idx,idx+2))
    gt = os.path.join(input_dir, '{}.jpg'.format(idx+1))
    data_list.extend([img0, img1, points14, points12, gt])
    images = [Image.open(pth) for pth in data_list]
    size = (384, 192)
    images = [transforms(img_.resize(size)).unsqueeze(0) for img_ in images]

    return images

def test(images, idx):
    print('Evaluating for {}'.format(idx))
    model.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():

        images = [img_.to(device) for img_ in images]
        points = torch.cat([images[2], images[3]], dim=1)
        out = model(images[0], images[1], points)

        imwrite(images[0], result_dir + '/{}csi.jpg'.format(idx))
        imwrite(images[1], result_dir + '/{}csi.jpg'.format(idx+2))
        imwrite(out[0], result_dir + '/{}csi.jpg'.format(idx+1))
    return

""" Entry Point """
def main(args):
    load_checkpoint(args, model, optimizer, save_loc+'/model_best1.pth')

    for i in range(1, 59, 2):
        images = read(i)
        test(images,i)

if __name__ == "__main__":
    main(args)
