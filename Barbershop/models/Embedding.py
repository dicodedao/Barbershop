import torch
from torch import nn
from Barbershop.models.Net import Net
import numpy as np
import os
from functools import partial
from Barbershop.utils.bicubic import BicubicDownSample
from Barbershop.datasets.image_dataset import ImagesDataset
from Barbershop.losses.embedding_loss import EmbeddingLossBuilder
from torch.utils.data import DataLoader
from tqdm import tqdm
import PIL
import torchvision
from Barbershop.utils.data_utils import convert_npy_code

toPIL = torchvision.transforms.ToPILImage()


class Embedding(nn.Module):
    def __init__(self, tool):
        super(Embedding, self).__init__()
        self.tool = tool
        self.loss_builder = EmbeddingLossBuilder(tool.opts)

    def setup_W_optimizer(self):

        opt_dict = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "sgdm": partial(torch.optim.SGD, momentum=0.9),
            "adamax": torch.optim.Adamax,
        }
        latent = []
        if self.tool.opts.tile_latent:
            tmp = self.tool.net.latent_avg.clone().detach().cuda()
            tmp.requires_grad = True
            for i in range(self.tool.net.layer_num):
                latent.append(tmp)
            optimizer_W = opt_dict[self.tool.opts.opt_name](
                [tmp], lr=self.tool.opts.learning_rate
            )
        else:
            for i in range(self.tool.net.layer_num):
                tmp = self.tool.net.latent_avg.clone().detach().cuda()
                tmp.requires_grad = True
                latent.append(tmp)
            optimizer_W = opt_dict[self.tool.opts.opt_name](
                latent, lr=self.tool.opts.learning_rate
            )

        return optimizer_W, latent

    def setup_FS_optimizer(self, latent_W, F_init):

        latent_F = F_init.clone().detach().requires_grad_(True)
        latent_S = []
        opt_dict = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "sgdm": partial(torch.optim.SGD, momentum=0.9),
            "adamax": torch.optim.Adamax,
        }
        for i in range(self.tool.net.layer_num):

            tmp = latent_W[0, i].clone()

            if i < self.tool.net.S_index:
                tmp.requires_grad = False
            else:
                tmp.requires_grad = True

            latent_S.append(tmp)

        optimizer_FS = opt_dict[self.tool.opts.opt_name](
            latent_S[self.tool.net.S_index :] + [latent_F], lr=self.tool.opts.learning_rate
        )

        return optimizer_FS, latent_F, latent_S

    def setup_dataloader(self, image_path=None):

        self.dataset = ImagesDataset(opts=self.tool.opts, image_path=image_path)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

    def check_W_existed(self, file_name, path_dict):
        dir = path_dict[file_name]
        w_file_name = f"{file_name}_w.npy"
        return os.path.exists(os.path.join(dir, w_file_name))

    def invert_images_in_W(self, image_path=None):
        self.setup_dataloader(image_path=image_path)
        device = self.tool.opts.device
        # ibar = tqdm(self.dataloader, desc="Images")

        path_dict = {}
        for p in image_path:
            file_name = os.path.splitext(os.path.basename(p))[0]
            path_dict[file_name] = os.path.dirname(p)

        for ref_im_H, ref_im_L, ref_name in self.dataloader:
            if self.check_W_existed(ref_name[0], path_dict):
                continue

            optimizer_W, latent = self.setup_W_optimizer()
            
            # pbar = tqdm(range(self.tool.opts.W_steps), desc="Embedding", leave=False)
            for step in range(self.tool.opts.W_steps):
                
                optimizer_W.zero_grad()
                latent_in = torch.stack(latent).unsqueeze(0)
                
                gen_im, _ = self.tool.net.generator(
                    [latent_in], input_is_latent=True, return_latents=False
                )
                im_dict = {
                    "ref_im_H": ref_im_H.to(device),
                    "ref_im_L": ref_im_L.to(device),
                    "gen_im_H": gen_im,
                    "gen_im_L": self.tool.downsample_256(gen_im),
                }

                loss, loss_dic = self.cal_loss(im_dict, latent_in)
                loss.backward()
                optimizer_W.step()

                # if self.tool.opts.verbose:
                #     pbar.set_description(
                #         "Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}".format(
                #             loss, loss_dic["l2"], loss_dic["percep"], loss_dic["p-norm"]
                #         )
                #     )
                
            
            self.save_W_results(ref_name, gen_im, latent_in, path_dict)

    def check_FS_existed(self, file_name, path_dict):
        dir = path_dict[file_name]
        fs_file_name = f"{file_name}_fs.npz"
        return os.path.exists(os.path.join(dir, fs_file_name))

    def invert_images_in_FS(self, image_path=None):
        self.setup_dataloader(image_path=image_path)
        device = self.tool.opts.device
        # ibar = tqdm(self.dataloader, desc="Images")

        path_dict = {}
        for p in image_path:
            file_name = os.path.splitext(os.path.basename(p))[0]
            path_dict[file_name] = os.path.dirname(p)

        for ref_im_H, ref_im_L, ref_name in self.dataloader:
            img_name = ref_name[0]
            if self.check_FS_existed(img_name, path_dict):
                continue

            latent_W_path = os.path.join(path_dict[img_name], f"{img_name}_w.npy")
            latent_W = torch.from_numpy(convert_npy_code(np.load(latent_W_path))).to(
                device
            )
            F_init, _ = self.tool.net.generator(
                [latent_W],
                input_is_latent=True,
                return_latents=False,
                start_layer=0,
                end_layer=3,
            )
            optimizer_FS, latent_F, latent_S = self.setup_FS_optimizer(latent_W, F_init)

            # pbar = tqdm(range(self.tool.opts.FS_steps), desc="Embedding", leave=False)
            for step in range(self.tool.opts.FS_steps):

                optimizer_FS.zero_grad()
                latent_in = torch.stack(latent_S).unsqueeze(0)
                gen_im, _ = self.tool.net.generator(
                    [latent_in],
                    input_is_latent=True,
                    return_latents=False,
                    start_layer=4,
                    end_layer=8,
                    layer_in=latent_F,
                )
                im_dict = {
                    "ref_im_H": ref_im_H.to(device),
                    "ref_im_L": ref_im_L.to(device),
                    "gen_im_H": gen_im,
                    "gen_im_L": self.tool.downsample_256(gen_im),
                }

                loss, loss_dic = self.cal_loss(im_dict, latent_in)
                loss.backward()
                optimizer_FS.step()

                # if self.tool.opts.verbose:
                #     pbar.set_description(
                #         "Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}, L_F loss: {:.3f}".format(
                #             loss,
                #             loss_dic["l2"],
                #             loss_dic["percep"],
                #             loss_dic["p-norm"],
                #             loss_dic["l_F"],
                #         )
                #     )

            self.save_FS_results(ref_name, gen_im, latent_in, latent_F, path_dict)

    def cal_loss(self, im_dict, latent_in, latent_F=None, F_init=None):
        loss, loss_dic = self.loss_builder(**im_dict)
        p_norm_loss = self.tool.net.cal_p_norm_loss(latent_in)
        loss_dic["p-norm"] = p_norm_loss
        loss += p_norm_loss

        if latent_F is not None and F_init is not None:
            l_F = self.tool.net.cal_l_F(latent_F, F_init)
            loss_dic["l_F"] = l_F
            loss += l_F

        return loss, loss_dic

    def save_W_results(self, ref_name, gen_im, latent_in, path_dict):
        save_latent = latent_in.detach().cpu().numpy()

        w_file_prefix = ref_name[0]
        latent_path = os.path.join(path_dict[w_file_prefix], f"{w_file_prefix}_w.npy")

        np.save(latent_path, save_latent)

    def save_FS_results(self, ref_name, gen_im, latent_in, latent_F, path_dict):

        fs_file_prefix = ref_name[0]

        latent_path = os.path.join(
            path_dict[fs_file_prefix], f"{fs_file_prefix}_fs.npz"
        )

        np.savez(
            latent_path,
            latent_in=latent_in.detach().cpu().numpy(),
            latent_F=latent_F.detach().cpu().numpy(),
        )

    def set_seed(self):
        if self.tool.opts.seed:
            torch.manual_seed(self.tool.opts.seed)
            torch.cuda.manual_seed(self.tool.opts.seed)
            torch.backends.cudnn.deterministic = True
