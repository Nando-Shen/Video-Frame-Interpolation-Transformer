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

result_dir = '/home/jiaming/flowdeep2rerun'

def read(data_root):
    data_list = []
    img0 = os.path.join(result_dir, data_root, 'flowdeep2.png')
    img1 = os.path.join(result_dir, data_root, 'frame3.png')

    points14 = os.path.join(result_dir, data_root, '121inter14.jpg')
    points12 = os.path.join(result_dir, data_root, '121inter12.jpg')
    gt = os.path.join(result_dir, data_root, 'frame2.jpg')
    data_list.extend([img0, img1, points14, points12, gt])
    images = [Image.open(pth) for pth in data_list]
    size = (384, 192)
    images = [transforms(img_.resize(size)).unsqueeze(0) for img_ in images]

    return images

def test(images, d):
    print('Evaluating for {}'.format(d))
    model.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():

        images = [img_.to(device) for img_ in images]
        points = torch.cat([images[2], images[3]], dim=1)
        out = model(images[0], images[1], points)

        imwrite(out[0], result_dir + '/' + d + '/flowdeep2_34.png')
    return

""" Entry Point """
def main(args):
    load_checkpoint(args, model, optimizer, save_loc+'/model_best1.pth')
    dirs = ['Disney_v4_12_001140_s2',
            'Disney_v4_13_002335_s2',
            'Disney_v4_21_029362_s2',
            'Japan_v2_1_003519_s2',
            'Japan_v2_1_130366_s3',
            'Disney_v4_20_006432_s2']
    for dir in dirs:
        images = read(dir)
        test(images,dir)

if __name__ == "__main__":
    main(args)
