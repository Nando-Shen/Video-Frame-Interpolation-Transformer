import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp


def joinTensors(X1 , X2 , type="concat"):

    if type == "concat":
        return torch.cat([X1 , X2] , dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1

class Conv_2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):
        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)

class upSplit(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.upconv = nn.ModuleList(
                [nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3,3), stride=(1,2,2), padding=1),
                 ]
            )
        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x, output_size):
        x = self.upconv[0](x, output_size=output_size)
        return x

class Conv_3d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    ]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, scale=1, c=64):
        super(IFBlock, self).__init__()
        self.scale = scale
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.conv1 = nn.ConvTranspose2d(c, 4, 4, 2, 1)

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=1. / self.scale, mode="bilinear", align_corners=False)
        x = self.conv0(x)
        x = self.convblock(x) + x
        x = self.conv1(x)
        flow = x
        if self.scale != 1:
            flow = F.interpolate(flow, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return flow


class IFNet(nn.Module):
    def __init__(self, args=None):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, scale=4, c=240)
        self.block1 = IFBlock(10, scale=2, c=150)
        self.block2 = IFBlock(10, scale=1, c=90)

    def forward(self, x):
        flow0 = self.block0(x)
        F1 = flow0
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = warp(x[:, :3], F1_large[:, :2])
        warped_img1 = warp(x[:, 3:], F1_large[:, 2:4])
        flow1 = self.block1(torch.cat((warped_img0, warped_img1, F1_large), 1))
        F2 = (flow0 + flow1)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = warp(x[:, :3], F2_large[:, :2])
        warped_img1 = warp(x[:, 3:], F2_large[:, 2:4])
        flow2 = self.block2(torch.cat((warped_img0, warped_img1, F2_large), 1))
        F3 = (flow0 + flow1 + flow2)

        return F3, [F1, F2, F3]


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class FlowRefineNet_Multis_Simple(nn.Module):
    def __init__(self, c=24, n_iters=1):
        super(FlowRefineNet_Multis_Simple, self).__init__()

        self.conv1 = Conv2(3, c, 1)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

    def forward(self, x0, x1, flow):
        bs = x0.size(0)

        inp = torch.cat([x0, x1], dim=0)
        s_1 = self.conv1(inp)  # 1
        s_2 = self.conv2(s_1)  # 1/2
        s_3 = self.conv3(s_2)  # 1/4
        s_4 = self.conv4(s_3)  # 1/8

        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.

        # warp features by the updated flow
        c0 = [s_1[:bs], s_2[:bs], s_3[:bs], s_4[:bs]]
        c1 = [s_1[bs:], s_2[bs:], s_3[bs:], s_4[bs:]]
        out0 = self.warp_fea(c0, flow[:, :2])
        out1 = self.warp_fea(c1, flow[:, 2:4])

        return flow, out0, out1

    def warp_fea(self, feas, flow):
        outs = []
        for i, fea in enumerate(feas):
            out = warp(fea, flow)
            outs.append(out)
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        return outs

class SKETCH(nn.Module):
    def __init__(self, args):
        super(SKETCH,self).__init__()
        c = 24
        self.flownet = IFNet()
        self.refinenet = FlowRefineNet_Multis_Simple(c=c, n_iters=1)


    def forward(self, img0, img1):

        B, _, H, W = img0.size()
        imgs = torch.cat((img0, img1), 1)
        flow, flow_list = self.flownet(imgs)
        flow, c0, c1 = self.refinenet(img0, img1, flow)

        out = warp(img0, flow[:, :2]/2)
        # out = img0[:, :1, :, :] - flow / 2
        # out = torch.nn.functional.grid_sample(input=img0, grid=flow, mode='bilinear', padding_mode='border',
        #                                 align_corners=True)
        # out = self.fuse(out)
        return out

if __name__ == '__main__':
    model = SKETCH('unet_18', n_inputs=4, n_outputs=1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('the number of network parameters: {}'.format(total_params))
    # inp = [torch.randn(1, 3, 225, 225).cuda() for i in range(4)]
    # out = model(inp)
    # print(out[0].shape, out[1].shape, out[2].shape)
