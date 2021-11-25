"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import hashlib
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

import fastmri
from fastmri import MriModule
from fastmri.data import transforms
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.models.MANet import MANet
import cv2
import matplotlib.pyplot as plt
import numpy as np

class SRModule(MriModule):
    """
    Unet training module.
    """

    def __init__(
        self,
        n_channels_in=1,
        n_channels_out=2,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        L=2,
        **kwargs,
    ):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net
                model.
            chans (int): Number of output channels of the first convolution
                layer.
            num_pool_layers (int): Number of down-sampling and up-sampling
                layers.
            drop_prob (float): Dropout probability.
            mask_type (str): Type of mask from ("random", "equispaced").
            center_fractions (list): Fraction of all samples to take from
                center (i.e., list of floats).
            accelerations (list): List of accelerations to apply (i.e., list
                of ints).
            lr (float): Learning rate.
            lr_step_size (int): Learning rate step size.
            lr_gamma (float): Learning rate gamma decay.
            weight_decay (float): Parameter for penalizing weights norm.
        """
        super().__init__(**kwargs)
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        # self.unet = UNet
        self.UNet_k = Dense_Unet_k(
            in_chan=2,
            out_chan=2,
            filters=64, 
            num_conv = 4,)
        self.UNet_img = Dense_Unet_img(
            in_chan=1,
            out_chan=1,
            filters=64, 
            num_conv = 4,)

    def forward(self, target_Kspace_T1,masked_Kspaces_T2):
        return self.UNet_k(target_Kspace_T1,masked_Kspaces_T2)
    def forward2(self, input_T1_img,input_T2_img):
        return self.UNet_img(input_T1_img,input_T2_img)

    def training_step(self, batch, batch_idx):
        #T1
        masked_Kspace_T1 = batch['masked_Kspaces_T1'].cuda().float()  
        target_Kspace_T1 = batch['target_Kspace_T1'].cuda().float()
        target_img_T1 = batch['target_img_T1'].cuda().float()
        maskedNot = batch['maskedNot'].cuda().float()
        #T2
        masked_Kspaces_T2 = batch['masked_Kspaces_T2'].cuda().float() 
        target_Kspace_T2 = batch['target_Kspace_T2'].cuda().float()
        target_img_T2 = batch['target_img_T2'].cuda().float()
        fname = batch['fname']
        slice_num = batch['slice_num']

        masked_img_T2 = self.inverseFT(masked_Kspaces_T2).cuda().float()
       
        output_T1k,output_T2k = self.forward(target_Kspace_T1,masked_Kspaces_T2)
        loss_T1 = F.l1_loss(output_T1k, target_Kspace_T1)
        loss_T2 = F.l1_loss(output_T2k, target_Kspace_T2)
        loss_k = 0.1*loss_T1+0.9*loss_T2

        input_T1_img = self.inverseFT(output_T1k)
        input_T2 = self.inverseFT(output_T2k)

        output_T1img,output_T2 = self.forward2(input_T1_img,input_T2)
        loss_T1img = F.l1_loss(output_T1img, target_img_T1)
        loss_T2img = F.l1_loss(output_T2, target_img_T2)
        loss_img = 0.1*loss_T1img+0.9*loss_T2img

        loss = loss_k + loss_img

        logs = {"loss": loss.detach()}

        return dict(loss=loss, log=logs)

    def inverseFT(self, Kspace):
        Kspace = Kspace.permute(0, 2, 3, 1)#last dimension=2
        img_cmplx = torch.ifft(Kspace, 2)
        img = torch.sqrt(img_cmplx[:, :, :, 0]**2 + img_cmplx[:, :, :, 1]**2)
        img = img[:, None, :, :]
        return img

    def contrastStretching(self,img, saturated_pixel=0.004):
        """ constrast stretching according to imageJ
        http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm"""
        values = np.sort(img, axis=None)
        nr_pixels = np.size(values)  # 像素数目
        lim = int(np.round(saturated_pixel*nr_pixels))
        v_min = values[lim]
        v_max = values[-lim-1]
        img = (img - v_min)*(255.0)/(v_max - v_min)
        img = np.minimum(255.0, np.maximum(0.0, img))  # 限制到0-255区间
        return img

    def fftshift(self, x, dim=None):

        if dim is None:
            dim = tuple(range(x.dim()))
            shift = [dim // 2 for dim in x.shape]
        elif isinstance(dim, int):
            shift = x.shape[dim] // 2
        else:
            shift = [x.shape[i] // 2 for i in dim]

        return roll(x, shift, dim)

    def imshow(self, img, title=""):
        """ Show image as grayscale. """
        if img.dtype == np.complex64 or img.dtype == np.complex128:
            print('img is complex! Take absolute value.')
            img = np.abs(img)

        plt.figure()
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.title(title)
        plt.show()


    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]

    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(img))

    def validation_step(self, batch, batch_idx):
        #T1
        masked_Kspace_T1 = batch['masked_Kspaces_T1'].cuda().float()  
        target_Kspace_T1 = batch['target_Kspace_T1'].cuda().float()
        target_img_T1 = batch['target_img_T1'].cuda().float()
        maskedNot = batch['maskedNot'].cuda().float()
        masks = batch['masks'].cuda().float()

        #T2
        masked_Kspaces_T2 = batch['masked_Kspaces_T2'].cuda().float() 
        target_Kspace_T2 = batch['target_Kspace_T2'].cuda().float()
        target_img_T2 = batch['target_img_T2'].cuda().float()
        fname = batch['fname']
        slice_num = batch['slice_num']

        masked_img_T2 = self.inverseFT(masked_Kspaces_T2).cuda().float()    

        output_T1k,output_T2k = self.forward(target_Kspace_T1,masked_Kspaces_T2)

        loss_T1 = F.l1_loss(output_T1k, target_Kspace_T1)
        loss_T2 = F.l1_loss(output_T2k, target_Kspace_T2)
        loss_k = 0.1*loss_T1+0.9*loss_T2


        input_T1img = self.inverseFT(output_T1k)#不是image
        input_T2 = self.inverseFT(output_T2k)#是mage

        output_T1img,output_T2 = self.forward2(input_T1img,input_T2)
        loss_T1img = F.l1_loss(output_T1img, target_img_T1)
        loss_T2img = F.l1_loss(output_T2, target_img_T2)
        loss_img = 0.1*loss_T1img+0.9*loss_T2img
        loss = loss_k + loss_img

        fnumber = torch.zeros(len(fname), dtype=torch.long, device=output_T2.device)
        for i, fn in enumerate(fname):
            fnumber[i] = (
                int(hashlib.sha256(fn.encode("utf-8")).hexdigest(), 16) % 10 ** 12
            )

        return {
            "fname": fnumber,
            "slice": slice_num,
            # "output": output * std + mean,
            # "target": target * std + mean,
            "output_T2": output_T2,
            "target_im_T2": target_img_T2,
            "val_loss": loss,
        }

    def test_step(self, batch, batch_idx):
        #T1
        masked_Kspace_T1 = batch['masked_Kspaces_T1'].cuda().float()#masked_kspace: torch.Size([1, 2, 256, 256])  
        target_Kspace_T1 = batch['target_Kspace_T1'].cuda().float()# target_kspace: torch.Size([1, 2, 256, 256])
        target_img_T1 = batch['target_img_T1'].cuda().float()#target_img: torch.Size([1, 1, 256, 256])
        #T2
        masked_Kspaces_T2 = batch['masked_Kspaces_T2'].cuda().float()#masked_kspace: torch.Size([1, 2, 256, 256])  
        target_Kspace_T2 = batch['target_Kspace_T2'].cuda().float()# target_kspace: torch.Size([1, 2, 256, 256])
        target_img_T2 = batch['target_img_T2'].cuda().float()#target_img: torch.Size([1, 1, 256, 256])

        masked_img_T2 = self.inverseFT(masked_Kspaces_T2).cuda().float()
        output_T1, output_T2 = self(target_img_T1,masked_img_T2)

        fname = batch['fname']
        slice_num = batch['slice_num']
        fnumber = torch.zeros(len(fname), dtype=torch.long, device=output_T2.device)
        for i, fn in enumerate(fname):
            fnumber[i] = (
                int(hashlib.sha256(fn.encode("utf-8")).hexdigest(), 16) % 10 ** 12
            )

        return {
            "fname": fnumber,
            "slice": slice_num,
            "output_T2": output_T2,
            "target_im_T2": target_img_T2,
            "test_loss": F.l1_loss(output_T2, target_img_T2),
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument("--in_chans", default=1, type=int)
        parser.add_argument("--out_chans", default=1, type=int)
        parser.add_argument("--chans", default=1, type=int)
        parser.add_argument("--num_pool_layers", default=4, type=int)
        parser.add_argument("--drop_prob", default=0.0, type=float)

        # data params
        
        # training params (opt)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--lr_step_size", default=40, type=int)
        parser.add_argument("--lr_gamma", default=0.1, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)

        parser.add_argument('--ixi-args', type=dict)

        return parser

