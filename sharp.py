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
from visualizer import get_local
get_local.activate()
from model.VFIformer_arch import VFIformerSmall
from dataset.atd12k import get_loader
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


def grid_show(to_shows, cols):
    rows = (len(to_shows) - 1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows * 8.5, cols * 2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()


def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()


def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)


def gray2rgb(image):
    return np.repeat(image[..., np.newaxis], 3, 2)


def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H / grid_size[0])
    delta_W = int(W / grid_size[1])

    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]

    padded_image = np.hstack((padding, image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W / 4), int(delta_H / 4)), 'CLS', fill=(0, 0, 0))  # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask), cls_weight)
    cls_weight = cls_weight / max(np.max(mask), cls_weight)

    if len(padding.shape) == 3:
        padding = padding[:, :, 0]
        padding[:, :] = np.min(mask)
    mask_to_pad = np.ones((1, 1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H, :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask

    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1], 4))
    meta_mask[delta_H:, 0: delta_W, :] = 1

    return padded_image, padded_mask, meta_mask


def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]

    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))

    padded_image, padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)

    if grid_index != 0:  # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index - 1) // grid_size[1]

    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1] + 1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')

    plt.savefig('vis.png')



def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    H, W = att_map.shape
    with_cls_token = False

    grid_image = highlight_grid(image, [grid_index], grid_size)

    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.show()


def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline='red', width=2)
    return image

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

# gt = gt.to(device)
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
# net = ResnetFeatureExtractor(model).to(device)

print(gt.size())
# gt_f = net(gt)
# print(gt_f.size())

target_layers = [model.module.final_fuse_block[2]]
# Note: input_tensor can be a batch tensor with several images!
img = torch.cat([images[0], images[1], images[2], images[3]], dim=1).to(device)

out = model(img)
cache = get_local.cache
print(list(cache.keys()))
attention_maps = cache['WindowCrossAttention.forward']
print(len(attention_maps))
visualize_grid_to_grid_with_cls(attention_maps[4][0,4,:,:], 105, gt)
# for name in model.state_dict():
#     print(name)
# print(list(model.modules()))
