import time

import torch
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image as imwrite
from PIL import Image

import config
import myutils
from loss import Loss
import shutil
import os

from model.VFIformer_arch import VFIformerSmall
from dataset.atd12k import get_loader

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)

class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])

    def __call__(self, x):
        return self.feature_extractor(x)[:, :, 0, 0]

def load_checkpoint(args, model, optimizer, path):
    print("loading checkpoint %s" % path)
    checkpoint = torch.load(path)
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = checkpoint.get("lr", args.lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

args, unparsed = config.get_args()
device = torch.device('cuda' if args.cuda else 'cpu')
args.device = device
args.resume_flownet = False
# train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)
test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
images, gt, imgpath = next(iter(test_loader))

gt = gt.to(device)
# img0 = os.path.join('image', 'frame1.jpg')
# img1 = os.path.join('image', 'frame3.jpg')
#
# points14 = os.path.join('image', 'inter14.jpg')
# points12 = os.path.join('image', 'inter12.jpg')
# points34 = os.path.join('image', 'inter34.jpg')
# gt = os.path.join('image', 'frame2.jpg')
# data_list = [img0, img1, points14, points12, points34, gt]
# images = [Image.open(pth) for pth in data_list]

model = VFIformerSmall(args)
model = torch.nn.DataParallel(model)
model.eval()

from torch.optim import Adamax
optimizer = Adamax(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
save_loc = os.path.join(args.checkpoint_dir, "checkpoints")

load_checkpoint(args, model, optimizer, save_loc + '/model_best1.pth')
net = ResnetFeatureExtractor(model).to(device)

print(gt.size())
gt_f = net(gt)
print(gt_f.size())

target_layers = [model.module.final_fuse_block[2]]
# Note: input_tensor can be a batch tensor with several images!
img = torch.cat([images[0], images[1], images[2], images[3]], dim=1).to(device)

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)

targets = [SimilarityToConceptTarget(gt_f)]

grayscale_cam = cam(input_tensor=img, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(gt, grayscale_cam, use_rgb=True)

for name in model.state_dict():
    print(name)
# print(list(model.modules()))
