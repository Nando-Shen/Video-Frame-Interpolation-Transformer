import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.pwc import PWC

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


class SKETCH(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.flownet = PWC()



    def forward(self, img0, img1):
        intWidth = img0.shape[2]
        intHeight = img0.shape[1]
        # tenPreprocessedOne = img0.view(1, 3, intHeight, intWidth)
        # tenPreprocessedTwo = img1.view(1, 3, intHeight, intWidth)
        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        tenPreprocessedOne = torch.nn.functional.interpolate(input=img0,
                                                             size=(intPreprocessedHeight, intPreprocessedWidth),
                                                             mode='bilinear', align_corners=False)
        tenPreprocessedTwo = torch.nn.functional.interpolate(input=img1,
                                                             size=(intPreprocessedHeight, intPreprocessedWidth),
                                                             mode='bilinear', align_corners=False)
        tenFlow = torch.nn.functional.interpolate(input=PWC(tenPreprocessedOne, tenPreprocessedTwo),
                                                  size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
        flow = tenFlow[0, :, :, :]
        print(flow.size())
        out = img0 - flow/2
        return out

if __name__ == '__main__':
    model = SKETCH('unet_18', n_inputs=4, n_outputs=1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('the number of network parameters: {}'.format(total_params))
    # inp = [torch.randn(1, 3, 225, 225).cuda() for i in range(4)]
    # out = model(inp)
    # print(out[0].shape, out[1].shape, out[2].shape)
