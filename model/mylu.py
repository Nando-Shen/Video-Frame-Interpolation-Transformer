import torch
from torch.autograd import Function
import torch.nn as nn
#
#
# class mylu(nn.Module):
#
#     def __init__(self):
#         super(mylu, self).__init__()
#         self.r = mylu_.apply  ### <-----注意此处
#
#     def forward(self, inputs):
#         outs = self.r(inputs)
#         return outs
#
#
# class mylu_(Function):
#     @staticmethod
#     def forward(ctx, inputs):
#         # output = inputs.new(inputs.size())
#         # output[inputs >= 0.] = 1
#         # output[inputs < 0.] = -1
#         ctx.save_for_backward(inputs)
#         return inputs.clamp(min=0)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input_, = ctx.saved_tensors
#         grad_output = grad_output.clone()
#         grad_output[input_ > 1.] *= 0.1
#         grad_output[input_ < 0.] *= 0.1
#         return grad_output

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        ctx.save_for_backward(input_)    # 将输入保存起来，在backward时使用
        output = input_.clamp(min=0, max=1)               # relu就是截断负数，让所有负数等于0
        return output
    @staticmethod
    def backward(ctx, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_outpu
        input_, = ctx.saved_tensors
        grad_output = grad_output.clone()
        grad_output[input_ > 1.] *= 0.1
        grad_output[input_ < 0.] *= 0.1
        return grad_output

def mylu(input_):
    # MyReLU()是创建一个MyReLU对象，
    # Function类利用了Python __call__操作，使得可以直接使用对象调用__call__制定的方法
    # __call__指定的方法是forward，因此下面这句MyReLU（）（input_）相当于
    # return MyReLU().forward(input_)
    return MyReLU().apply(input_)
