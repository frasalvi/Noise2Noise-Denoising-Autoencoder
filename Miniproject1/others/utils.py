# -*- coding: utf-8 -*-

import torch

def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()

def evaluate_psnr(net, validation_noisy_imgs, validation_clean_imgs):
    denoised = net.predict(validation_noisy_imgs)
    return compute_psnr(denoised/255,validation_clean_imgs/255)