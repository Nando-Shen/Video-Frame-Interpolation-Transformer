import torch
from torch.autograd import Function
import torch.nn as nn


class mylu(nn.Module):

    def __init__(self, *kargs, **kwargs):
        super(mylu_, self).__init__(*kargs, **kwargs)
        self.r = mylu_.apply  ### <-----注意此处

    def forward(self, inputs):
        outs = self.r(inputs)
        return outs


class mylu_(Function):
    @staticmethod
    def forward(ctx, inputs):
        # output = inputs.new(inputs.size())
        # output[inputs >= 0.] = 1
        # output[inputs < 0.] = -1
        ctx.save_for_backward(inputs)
        return inputs.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_output = grad_output.clone()
        grad_output[input_ > 1.] *= 0.1
        grad_output[input_ < 0.] *= 0.1
        return grad_output
