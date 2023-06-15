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
from Barbershop.models.Alignment import Alignment
from Barbershop.models.Blending import Blending

from Barbershop.main import *


class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ------ Received request")

        file_obj = request.FILES.get("file")
        target = request.POST.get("target")  # from 1 -> 8
        auto_crop = request.POST.get("auto_crop").lower() in ("yes", "true", "t", "1")

        storage = FileSystemStorage()
        file_ext = file_obj.name.split(".")[-1]
        file_name = uuid.uuid1().hex
        uploaded_file_path = os.path.join(settings.TOOL.opts.input_dir, f'{file_name}.{file_ext}')
        storage.save(uploaded_file_path, file_obj)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ------ Stored original file")

        if auto_crop:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] ------ Begin detect and align face to center of image"
            )

            valid_face = crop_image(settings.TOOL, uploaded_file_path)

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] ------ Finish detect and align face to center of image"
            )
        else:
            valid_face = True

        if not valid_face:
            return Response(
                {"status": "Can not find a face in the input image"},
                status=status.HTTP_406_NOT_ACCEPTABLE,
            )

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] ------ Begin transfer hair style"
        )

        im_path2 = os.path.join(settings.TOOL.opts.template_dir, f"{target}.png")
        
        transferHair(settings.TOOL, uploaded_file_path, im_path2, im_path2)

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] ------ Finish transfer hair style"
        )

        #clean up input file
        for f in glob.glob(os.path.join(settings.TOOL.opts.input_dir, file_name + '*')):
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