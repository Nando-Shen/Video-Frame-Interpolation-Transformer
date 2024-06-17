import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image as imwrite

import config
import myutils
from loss import Loss
import shutil

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

if args.dataset == "vimeo90K_septuplet":
    from dataset.vimeo90k_septuplet import get_loader
    train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == 'atd12k':
    from dataset.atd12k import get_loader
    train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == 'animerun':
    from dataset.animerun import get_loader
    train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
else:
    raise NotImplementedError

if args.model == 'VFIT_S':
    from model.VFIT_S import UNet_3D_3D
elif args.model == 'VFIT_B':
    from model.VFIT_B import UNet_3D_3D
elif args.model == 'VFI':
    from model.VFIformer_arch import VFIformer

print("Building model: %s"%args.model)
if args.model == 'VFI':
    args.device = device
    args.resume_flownet = False
    model = VFIformer(args)
else:
    model = UNet_3D_3D( n_inputs=args.nbr_frame, joinType=args.joinType)
model = torch.nn.DataParallel(model).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('the number of network parameters: {}'.format(total_params))

##### Define Loss & Optimizer #####
criterion = Loss(args)

## ToDo: Different learning rate schemes for different parameters
from torch.optim import Adamax
optimizer = Adamax(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

scaler = GradScaler()

def train(args, epoch):
    torch.cuda.empty_cache()
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.train()
    criterion.train()

    for i, (images, gt_image, flow) in enumerate(train_loader):

        # Build input batch
        images = [img_.to(device) for img_ in images]
        points = torch.cat([images[2]], 1)

        # Forward
        optimizer.zero_grad()
        if args.model == 'VFI':
            with autocast():
                out = model(images[0], images[1], points, flow)
                gt = gt_image.to(device)

                loss, _ = criterion(out, gt)
                overall_loss = loss

                losses['total'].update(loss.item())
        else:
            out_ll, out_l, out = model(images)
            gt = gt_image.to(device)

            loss, _ = criterion(out, gt)
            overall_loss = loss

            losses['total'].update(loss.item())
            overall_loss.backward()
            optimizer.step()

        scaler.scale(overall_loss).backward()
        scaler.step(optimizer)

        scaler.update()

        # Calc metrics & print logs
        if i % args.log_iter == 0:
            with autocast():
                myutils.eval_metrics(out, gt, psnrs, ssims)

            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tPSNR: {:.4f}  Lr:{:.6f}'.format(
                epoch, i, len(train_loader), losses['total'].avg, psnrs.avg , optimizer.param_groups[0]['lr'], flush=True))

            # Reset metrics
            losses, psnrs, ssims = myutils.init_meters(args.loss)


def test(args, epoch):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()
    criterion.eval()
    torch.cuda.empty_cache()

    t = time.time()
    with torch.no_grad():
        for i, (images, gt_image, datapath, flow) in enumerate(tqdm(test_loader)):

            images = [img_.to(device) for img_ in images]
            points = torch.cat([images[2]], dim=1)
            if args.model == 'VFI':
                out = model(images[0], images[1], points, flow)
            else:
                out = model(images)

            gt = gt_image.to(device)

            # print(out.size())

            # out = model(images) ## images is a list of neighboring frames
            # for idx in range(out.size()[0]):
                # print(idx)
                # print(datapath[idx])
                # os.makedirs(args.result_dir + '/' + datapath[idx])
                # imwrite(out[idx], args.result_dir + '/' + datapath[idx] + '/flowdeep2.png')

            # Save loss values
            loss, loss_specific = criterion(out, gt)
            for k, v in losses.items():
                if k != 'total':
                    v.update(loss_specific[k].item())
            losses['total'].update(loss.item())

            # Evaluate metrics
            myutils.eval_metrics(out, gt, psnrs, ssims)

    return losses['total'].avg, psnrs.avg, ssims.avg

def testt(args, epoch):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()
    criterion.eval()
    torch.cuda.empty_cache()

    t = time.time()
    with torch.no_grad():
        for i, (images, gt_image, datapath) in enumerate(tqdm(test_loader)):

            images = [img_.to(device) for img_ in images]
            if args.model == 'VFI':
                out, flow_list = model(images[0], images[1])
                gt = gt_image.to(device)
            else:
                out = model(images)
                gt = gt_image.to(device)

            print(out.size())

            for idx in range(out.size()[0]):
            #     # print(idx)
            #     # print(datapath[idx])
                os.makedirs(args.result_dir + '/' + datapath[idx])
                imwrite(out[idx], args.result_dir + '/' + datapath[idx] + '/vfit-animerun.png')


            # out = model(images) ## images is a list of neighboring frames

            # Save loss values
            loss, loss_specific = criterion(out, gt)
            for k, v in losses.items():
                if k != 'total':
                    v.update(loss_specific[k].item())
            losses['total'].update(loss.item())

            # Evaluate metrics
            myutils.eval_metrics(out, gt, psnrs, ssims)

    return losses['total'].avg, psnrs.avg, ssims.avg


def print_log(epoch, num_epochs, one_epoch_time, oup_pnsr, oup_ssim, Lr):
    print('({0:.0f}s) Epoch [{1}/{2}], Val_PSNR:{3:.2f}, Val_SSIM:{4:.4f}'
          .format(one_epoch_time, epoch, num_epochs, oup_pnsr, oup_ssim))
    # write training log
    with open('./training_log/train_log.txt', 'a') as f:
        print(
            'Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}, Lr:{6}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    one_epoch_time, epoch, num_epochs, oup_pnsr, oup_ssim, Lr), file=f)

lr_schular = [2e-4, 1e-4, 5e-5, 2.5e-5, 5e-6, 1e-6]
training_schedule = [40, 60, 75, 85, 95, 100]

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for i in range(len(training_schedule)):
        if epoch < training_schedule[i]:
            current_learning_rate = lr_schular[i]
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_learning_rate
        print('Learning rate sets to {}.'.format(param_group['lr']))

""" Entry Point """
def main(args):
    # load_checkpoint(args, model, optimizer, save_loc+'/model_best1.pth')
    # test_loss, psnr, ssim = testt(args, args.start_epoch)
    # print("psnr :{}, ssim:{}".format(psnr, ssim))
    # exit()
    load_checkpoint(args, model, optimizer, '/home/kuhu6123/jshe2377/VFIformer/Video-Frame-Interpolation-Transformer/ckp/checkpoints/model_best.pth')
    test_loss, psnr, ssim = testt(args, args.start_epoch)
    print("psnr :{}, ssim:{}".format(psnr, ssim))
    exit()

    best_psnr = 0
    for epoch in range(args.start_epoch, args.max_epoch):
        adjust_learning_rate(optimizer, epoch)
        start_time = time.time()
        train(args, epoch)

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': optimizer.param_groups[-1]['lr']
        }, os.path.join(save_loc, 'checkpoint.pth'))

        test_loss, psnr, ssim = test(args, epoch)

        # save checkpoint
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        if is_best:
            shutil.copyfile(os.path.join(save_loc, 'checkpoint.pth'), os.path.join(save_loc, 'model_best.pth'))

        one_epoch_time = time.time() - start_time
        print_log(epoch, args.max_epoch, one_epoch_time, psnr, ssim, optimizer.param_groups[-1]['lr'])

if __name__ == "__main__":
    main(args)

