import math
import torch
import torch.nn as nn
from functools import partial


def calculate(model, blockID, ratio):
    flops = [0]


    def ops_conv(self, input, flops):
        batch_size = input[0].size(0)
        input_planes = input[0].size(1)
        input_height = input[0].size(2)
        input_width = input[0].size(3)

        groups = self.groups or 1
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (input_planes / float(groups)) * 2
        bias_ops = self.bias is not None and 1 or 0
        ops_per_element = kernel_ops + bias_ops

        output_width = math.floor((input_width + 2 * self.padding[0] - self.kernel_size[0]) / float(self.stride[0]) + 1)
        output_height = math.floor((input_height + 2 * self.padding[1] - self.kernel_size[1]) / float(self.stride[0]) + 1)

        flops[0] += batch_size * self.out_channels * output_width * output_height * ops_per_element


    def ops_pooling(self, input, flops):
        batch_size = input[0].size(0)
        input_planes = input[0].size(1)
        input_height = input[0].size(2)
        input_width = input[0].size(3)
        kernel_size = ('int' in str(type(self.kernel_size))) and [self.kernel_size, self.kernel_size] or self.kernel_size
        kernel_ops = kernel_size[0] * kernel_size[1]
        stride = ('int' in str(type(self.stride))) and [self.stride, self.stride] or self.stride
        padding = ('int' in str(type(self.padding))) and [self.padding, self.padding] or self.padding

        output_width = math.floor((input_width + 2 * padding[0] - kernel_size[0]) / float(stride[0]) + 1)
        output_height = math.floor((input_height + 2 * padding[1] - kernel_size[1]) / float(stride[0]) + 1)

        flops[0] += batch_size * input_planes * output_width * output_height * kernel_ops


    def ops_adaptivepooling(self, input, output, flops):
        batch_size = input[0].size(0)
        input_planes = input[0].size(1)
        input_height = input[0].size(2)
        input_width = input[0].size(3)

        for i in range(output.size(2)):
            y_start = int(math.floor(float(i * input_height) / output.size(2)))
            y_end =  int(math.ceil(float((i + 1) * input_height) / output.size(2)))
            for j in range(output.size(3)):
                x_start = int(math.floor(float(j * input_width) / output.size(3)))
                x_end =  int(math.ceil(float((j + 1) * input_width) / output.size(3)))

                flops[0] += batch_size * input_planes * (y_end-y_start+1) * (x_end-x_start+1)


    def ops_linear(self, input, flops):
        batch_size = input[0].dim() == 2 and input[0].size(0) or 1
        weight_ops = self.weight.nelement() * 2
        bias_ops = self.bias is None and 0 or self.bias.nelement()
        ops_per_sample = weight_ops + bias_ops

        flops[0] += batch_size * ops_per_sample


    def ops_batchnorm(self, input, flops):
        flops[0] += input[0].nelement() * 2


    def ops_relu(self, input, flops):
        flops[0] += input[0].nelement()


    def ops_residual_addition(self, input, output, flops):
        flops[0] += output[0].nelement()


    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_pre_hook(partial(ops_conv, flops=flops)))
        elif isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_pre_hook(partial(ops_batchnorm, flops=flops)))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_pre_hook(partial(ops_linear, flops=flops)))
        elif isinstance(m, nn.ReLU):
            hooks.append(m.register_forward_pre_hook(partial(ops_relu, flops=flops)))
        elif isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AvgPool2d):
            hooks.append(m.register_forward_pre_hook(partial(ops_pooling, flops=flops)))
        elif isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.AdaptiveMaxPool2d):
            hooks.append(m.register_forward_hook(partial(ops_adaptivepooling, flops=flops)))
        elif 'Bottleneck' in str(type(m)):
            hooks.append(m.register_forward_hook(partial(ops_residual_addition, flops=flops)))

    try:
        model(torch.rand(1,3,224,224).float(), blockID, ratio)
    except:
        flops[0] = 0
        model(torch.rand(1,3,224,224).float().cuda(), blockID, ratio)

    for h in hooks:
        h.remove()
    del hooks[:]

    return int(flops[0])