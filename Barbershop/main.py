import argparse
from collections import namedtuple


import torch
import numpy as np
import sys
import os
import dlib
from django.conf import settings


from PIL import Image


from Barbershop.models.Embedding import Embedding
from Barbershop.models.Alignment import Alignment
from Barbershop.models.Blending import Blending
from Barbershop.models.Tool import Tool

from Barbershop.utils.shape_predictor import align_face
import uuid

from datetime import datetime

def preload_model():
    default_args = {
        "input_dir": f"{settings.MEDIA_ROOT}/user_input",
        "output_dir": f"{settings.MEDIA_ROOT}/user_output",
        "template_dir": f"{settings.MEDIA_ROOT}/template",
        "sign": "realistic",
        "smooth": 5,
        "size": 1024,
        "ckpt": f"{settings.BASE_DIR}/Barbershop/pretrained_models/ffhq.pt",
        "channel_multiplier": 2,
        "latent": 512,
        "n_mlp": 8,
        "device": "cuda",
        "seed": None,
        "tile_latent": True,
        "opt_name": "adam",
        "learning_rate": 0.03,
        "lr_schedule": "fixed",
        "save_intermediate": False,
        "save_interval": 300,
        "verbose": False,
        "seg_ckpt": f"{settings.BASE_DIR}/Barbershop/pretrained_models/seg.pth",
        "shape_predictor": f"{settings.BASE_DIR}/Barbershop/pretrained_models/shape_predictor_68_face_landmarks.dat",
        "percept_lambda": 1.0,
        "l2_lambda": 1.0,
        "p_norm_lambda": 0.001,
        "l_F_lambda": 0.1,
        "W_steps": 100,
        "FS_steps": 100,
        "ce_lambda": 1.0,
        "style_lambda": 4e4,
        "align_steps1": 70,
        "align_steps2": 50,
        "face_lambda": 1.0,
        "hair_lambda": 1.0,
        "blend_steps": 20,
    }
    ArgsTuple = namedtuple("ArgsTuple", default_args)
    args = ArgsTuple(**default_args)
    return Tool(args)

def transferHair(tool, im_path1, im_path2, im_path3):
    embedding = Embedding(tool)
    im_set = {im_path1, im_path2, im_path3}
    embedding.invert_images_in_W([*im_set])
    embedding.invert_images_in_FS([*im_set])

    align = Alignment(tool)
    align.align_images(
        im_path1, 
        im_path2, 
        sign=tool.opts.sign, 
        align_more_region=False, 
        smooth=tool.opts.smooth,
    )
    if im_path2 != im_path3:
        align.align_images(
            im_path1,
            im_path3,
            sign=tool.opts.sign,
            align_more_region=False,
            smooth=tool.opts.smooth,
            save_intermediate=False,
        )

    blend = Blending(tool)
    blend.blend_images(im_path1, im_path2, im_path3, sign=tool.opts.sign)

def crop_image(tool, im_path):
    faces = align_face(im_path, tool.shape_predictor)
    if len(faces) > 0:
        new_im_path = os.path.join(tool.opts.input_dir, f"{uuid.uuid1().hex}.png")
        faces[0].save(new_im_path)
        return new_im_path

