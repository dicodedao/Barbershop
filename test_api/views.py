from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

from django.core.files.storage import FileSystemStorage
from django.core.files import File
from django.http import FileResponse
from django.conf import settings
from collections import namedtuple

import barbershop_api.settings as settings
import uuid
import subprocess
import os, glob
from datetime import datetime

import requests

from Barbershop.models.Alignment import Alignment
from Barbershop.models.Blending import Blending

from Barbershop.main import *

def transferHairStyle(file_name, file_ext, target, auto_crop,):
    input_file_path = os.path.join(settings.TOOL.opts.input_dir, f'{file_name}.{file_ext}')

    if auto_crop:
        valid_face = crop_image(settings.TOOL, input_file_path)
    else:
        valid_face = True

    if not valid_face:
        return Response(
            {"status": "Can not find a face in the input image"},
            status=status.HTTP_406_NOT_ACCEPTABLE,
        )

    target_file_path = os.path.join(settings.TOOL.opts.template_dir, f"{target}.png")
    
    transferHair(settings.TOOL, input_file_path, target_file_path, target_file_path)

    # clean up input files
    for f in glob.glob(os.path.join(settings.TOOL.opts.input_dir, f'*{file_name}*')):
        os.remove(f)

    result_path = os.path.join(
        settings.TOOL.opts.output_dir,
        f'{file_name}_result.png',
    )
    if os.path.exists(result_path):
        return FileResponse(open(result_path, "rb"))
    else:
        return Response(
            {"status": "Hair transfer failed"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get("file")
        target = request.POST.get("target")  # from 1 -> 8
        auto_crop = request.POST.get("auto_crop").lower() in ("yes", "true", "t", "1")

        storage = FileSystemStorage()
        file_ext = file_obj.name.split(".")[-1]
        file_name = uuid.uuid1().hex
        uploaded_file_path = os.path.join(settings.TOOL.opts.input_dir, f'{file_name}.{file_ext}')
        storage.save(uploaded_file_path, file_obj)

        return transferHairStyle(file_name, file_ext, target, auto_crop)
        
class TransferView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):

        file_url = request.POST.get("file_url")
        target = request.POST.get("target")  # from 1 -> 8
        auto_crop = request.POST.get("auto_crop").lower() in ("yes", "true", "t", "1")

        file_ext = file_url.split('.')[-1]
        file_name = uuid.uuid1().hex
        uploaded_file_path = os.path.join(settings.TOOL.opts.input_dir, f'{file_name}.{file_ext}')
        
        #download file
        file_data = requests.get(file_url).content
        with open(uploaded_file_path, 'wb') as handler:
            handler.write(file_data)

        return transferHairStyle(file_name, file_ext, target, auto_crop)
        

class CreateTemplateView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get("file")
        file_id = request.POST.get('id')
        
        uploaded_file_path = os.path.join(settings.TOOL.opts.template_dir, f'{file_id}.png')
        if os.path.exists(uploaded_file_path):
            os.remove(uploaded_file_path)

        w_path = os.path.join(settings.TOOL.opts.template_dir, f'{file_id}_w.npy')
        if os.path.exists(w_path):
            os.remove(w_path)

        fs_path = os.path.join(settings.TOOL.opts.template_dir, f'{file_id}_fs.npz')
        if os.path.exists(fs_path):
            os.remove(fs_path)

        storage = FileSystemStorage()
        storage.save(uploaded_file_path, file_obj)
        
        
        if crop_image(settings.TOOL, uploaded_file_path):
            return Response({"status": "Success"})
        else:
            os.remove(uploaded_file_path)
            return Response(
                {"status": "Can not find a face in the input image"},
                status=status.HTTP_406_NOT_ACCEPTABLE,
            )
        