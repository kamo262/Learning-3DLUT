import math
from itertools import product

import torch
import torch.nn as nn
import numpy as np

import trilinear


class LUT_3D(nn.Module):
    def __init__(self, dim=33):
        super(LUT_3D, self).__init__()

        file = open(f"IdentityLUT{dim}.txt", "r")
        lines = file.readlines()
        buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)

        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    n = i * dim * dim + j * dim + k
                    x = lines[n].split()
                    buffer[0, i, j, k] = float(x[0])
                    buffer[1, i, j, k] = float(x[1])
                    buffer[2, i, j, k] = float(x[2])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        return output


class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim**3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        assert 1 == trilinear.forward(lut, x, output, dim, shift, binsize, W, H, batch)

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]

        ctx.save_for_backward(*variables)

        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):

        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])

        assert 1 == trilinear.backward(
            x, x_grad, lut_grad, dim, shift, binsize, W, H, batch
        )
        return lut_grad, x_grad


class TrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)


class TV_3D(nn.Module):
    def __init__(self, dim=33, mn_margin=0.0, tv_type="original"):
        super(TV_3D, self).__init__()

        self.mn_margin = mn_margin
        self.tv_type = tv_type

        self.relu = torch.nn.ReLU()

        file = open(f"IdentityLUT{dim}.txt", "r")
        lines = file.readlines()
        buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)

        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    n = i * dim * dim + j * dim + k
                    x = lines[n].split()
                    buffer[0, i, j, k] = float(x[0])
                    buffer[1, i, j, k] = float(x[1])
                    buffer[2, i, j, k] = float(x[2])
        self.identity_LUT = torch.from_numpy(buffer)

    def forward(self, LUT):

        dif_r = LUT.LUT[:, :, :, :-1] - LUT.LUT[:, :, :, 1:]
        dif_g = LUT.LUT[:, :, :-1, :] - LUT.LUT[:, :, 1:, :]
        dif_b = LUT.LUT[:, :-1, :, :] - LUT.LUT[:, 1:, :, :]
        if self.tv_type == "original":
            tv = (
                torch.mean(dif_r**2) + torch.mean(dif_g**2) + torch.mean(dif_b**2)
            )
        else:
            tv = (
                torch.mean(dif_r[1:] ** 2)
                + torch.mean(dif_g[[0, 2]] ** 2)
                + torch.mean(dif_b[:2] ** 2)
            )

        mn = (
            torch.mean(self.relu(dif_r[0] + self.mn_margin))
            + torch.mean(self.relu(dif_g[1] + self.mn_margin))
            + torch.mean(self.relu(dif_b[2] + self.mn_margin))
        )

        dif_r = self.identity_LUT[0] - LUT.LUT[0]
        dif_g = self.identity_LUT[1] - LUT.LUT[1]
        dif_b = self.identity_LUT[2] - LUT.LUT[2]
        identity = (
            torch.mean(dif_r**2) + torch.mean(dif_g**2) + torch.mean(dif_b**2)
        )

        return tv, mn, identity


def trilinear_forward(lut, image, output, dim, shift, binsize, width, height, channels):
    output_size = height * width

    for index in range(output_size):
        r = image[index]
        g = image[index + width * height]
        b = image[index + width * height * 2]

        r_id = math.floor(r / binsize)
        g_id = math.floor(g / binsize)
        b_id = math.floor(b / binsize)

        r_d = math.fmod(r, binsize) / binsize
        g_d = math.fmod(g, binsize) / binsize
        b_d = math.fmod(b, binsize) / binsize

        id000 = r_id + g_id * dim + b_id * dim * dim
        id100 = r_id + 1 + g_id * dim + b_id * dim * dim
        id010 = r_id + (g_id + 1) * dim + b_id * dim * dim
        id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim
        id001 = r_id + g_id * dim + (b_id + 1) * dim * dim
        id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim
        id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim
        id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim

        w000 = (1 - r_d) * (1 - g_d) * (1 - b_d)
        w100 = r_d * (1 - g_d) * (1 - b_d)
        w010 = (1 - r_d) * g_d * (1 - b_d)
        w110 = r_d * g_d * (1 - b_d)
        w001 = (1 - r_d) * (1 - g_d) * b_d
        w101 = r_d * (1 - g_d) * b_d
        w011 = (1 - r_d) * g_d * b_d
        w111 = r_d * g_d * b_d

        output[index] = (
            w000 * lut[id000]
            + w100 * lut[id100]
            + w010 * lut[id010]
            + w110 * lut[id110]
            + w001 * lut[id001]
            + w101 * lut[id101]
            + w011 * lut[id011]
            + w111 * lut[id111]
        )

        output[index + width * height] = (
            w000 * lut[id000 + shift]
            + w100 * lut[id100 + shift]
            + w010 * lut[id010 + shift]
            + w110 * lut[id110 + shift]
            + w001 * lut[id001 + shift]
            + w101 * lut[id101 + shift]
            + w011 * lut[id011 + shift]
            + w111 * lut[id111 + shift]
        )

        output[index + width * height * 2] = (
            w000 * lut[id000 + shift * 2]
            + w100 * lut[id100 + shift * 2]
            + w010 * lut[id010 + shift * 2]
            + w110 * lut[id110 + shift * 2]
            + w001 * lut[id001 + shift * 2]
            + w101 * lut[id101 + shift * 2]
            + w011 * lut[id011 + shift * 2]
            + w111 * lut[id111 + shift * 2]
        )

    return output
