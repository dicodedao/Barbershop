import torch
from torch import nn
from Barbershop.models.Net import Net
import numpy as np
import os
from Barbershop.utils.bicubic import BicubicDownSample
from tqdm import tqdm
import PIL
import torchvision
from Barbershop.models.face_parsing.model import BiSeNet, seg_mean, seg_std
from Barbershop.models.optimizer.ClampOptimizer import ClampOptimizer
from Barbershop.losses.blend_loss import BlendLossBuilder
import torch.nn.functional as F
import cv2
from Barbershop.utils.data_utils import load_FS_latent
from Barbershop.utils.data_utils import cuda_unsqueeze
from Barbershop.utils.image_utils import (
    load_image,
    dilate_erosion_mask_path,
    dilate_erosion_mask_tensor,
)
from Barbershop.utils.model_utils import download_weight

toPIL = torchvision.transforms.ToPILImage()


class Blending(nn.Module):
    def __init__(self, tool):
        super(Blending, self).__init__()
        self.tool = tool
        self.loss_builder = BlendLossBuilder(tool.opts)

    def setup_blend_optimizer(self):

        interpolation_latent = torch.zeros(
            (self.tool.net.layer_num, 512), requires_grad=True, device=self.tool.opts.device
        )

        opt_blend = ClampOptimizer(
            torch.optim.Adam, [interpolation_latent], lr=self.tool.opts.learning_rate
        )

        return opt_blend, interpolation_latent

    def blend_images(self, img_path1, img_path2, img_path3, sign="realistic"):

        device = self.tool.opts.device
        input_dir = self.tool.opts.input_dir

        im_name_1 = os.path.splitext(os.path.basename(img_path1))[0]
        im_name_2 = os.path.splitext(os.path.basename(img_path2))[0]
        im_name_3 = os.path.splitext(os.path.basename(img_path3))[0]

        I_1 = load_image(img_path1, downsample=True).to(device).unsqueeze(0)
        I_3 = load_image(img_path3, downsample=True).to(device).unsqueeze(0)

        HM_1D, _ = cuda_unsqueeze(dilate_erosion_mask_path(img_path1, self.tool.seg), device)
        HM_3D, HM_3E = cuda_unsqueeze(
            dilate_erosion_mask_path(img_path3, self.tool.seg), device
        )

        opt_blend, interpolation_latent = self.setup_blend_optimizer()
        latent_1, latent_F_mixed = load_FS_latent(
            os.path.join(input_dir, "align_{}_{}.npz".format(im_name_1, im_name_3)),
            device,
        )
        latent_3, _ = load_FS_latent(os.path.splitext(img_path3)[0] + "_fs.npz", device)

        with torch.no_grad():
            I_X, _ = self.tool.net.generator(
                [latent_1],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F_mixed,
            )
            I_X_0_1 = (I_X + 1) / 2
            IM = (self.tool.downsample(I_X_0_1) - seg_mean) / seg_std
            down_seg, _, _ = self.tool.seg(IM)
            current_mask = torch.argmax(down_seg, dim=1).long().cpu().float()
            HM_X = torch.where(
                current_mask == 10,
                torch.ones_like(current_mask),
                torch.zeros_like(current_mask),
            )
            HM_X = F.interpolate(
                HM_X.unsqueeze(0), size=(256, 256), mode="nearest"
            ).squeeze()
            HM_XD, _ = cuda_unsqueeze(dilate_erosion_mask_tensor(HM_X), device)
            target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        # pbar = tqdm(range(self.tool.opts.blend_steps), desc="Blend", leave=False)
        for step in range(self.tool.opts.blend_steps):

            opt_blend.zero_grad()

            latent_mixed = latent_1 + interpolation_latent.unsqueeze(0) * (
                latent_3 - latent_1
            )

            I_G, _ = self.tool.net.generator(
                [latent_mixed],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F_mixed,
            )
            I_G_0_1 = (I_G + 1) / 2

            im_dict = {
                "gen_im": self.tool.downsample_256(I_G),
                "im_1": I_1,
                "im_3": I_3,
                "mask_face": target_mask,
                "mask_hair": HM_3E,
            }
            loss, loss_dic = self.loss_builder(**im_dict)

            # if self.tool.opts.verbose:
            #     pbar.set_description(
            #         "Blend Loss: {:.3f}, face: {:.3f}, hair: {:.3f}".format(
            #             loss, loss_dic["face"], loss_dic["hair"]
            #         )
            #     )

            loss.backward()
            opt_blend.step()

        ############## Load F code from  '{}_{}.npz'.format(im_name_1, im_name_2)
        _, latent_F_mixed = load_FS_latent(
            os.path.join(input_dir, "align_{}_{}.npz".format(im_name_1, im_name_2)),
            device,
        )
        I_G, _ = self.tool.net.generator(
            [latent_mixed],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_F_mixed,
        )

        self.save_blend_results(
            im_name_1,
            I_G,
        )

    def save_blend_results(
        self,
        im_name_1,
        gen_im,
    ):
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        output_image_path = os.path.join(
            self.tool.opts.output_dir, f"{im_name_1}_result.png"
        )
        save_im.save(output_image_path)
