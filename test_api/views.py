from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

from django.core.files.storage import FileSystemStorage
from django.core.files import File
from django.http import FileResponse
from django.conf import settings

import barbershop_api.settings as settings
import uuid
import subprocess
import os
from datetime import datetime
from Barbershop.models.Alignment import Alignment
from Barbershop.models.Blending import Blending


class FileUploadView2(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ------ Received request")

        file_obj = request.FILES.get("file")
        target = request.POST.get("target")  # from 1 -> 8
        auto_crop = request.POST.get("auto_crop").lower() in ("yes", "true", "t", "1")
        print(f"AUTO_CROP: {auto_crop}")

        storage = FileSystemStorage()
        file_ext = file_obj.name.split(".")[-1]
        uploaded_file = f"{uuid.uuid1().hex}.{file_ext}"
        storage.save(
            os.path.join(settings.MEDIA_ROOT, "user_input", uploaded_file), file_obj
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ------ Stored original file")

        aligned_file = uploaded_file
        if auto_crop:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] ------ Begin detect and align face to center of image"
            )
            aligned_file = f"{uuid.uuid1().hex}.png"
            align_cmd = [
                "python",
                f"{settings.BASE_DIR}/Barbershop/align_face.py",
                "--input",
                f"{settings.MEDIA_ROOT}/user_input/{uploaded_file}",
                "--output",
                f"{settings.MEDIA_ROOT}/user_input/{aligned_file}",
                "--shape_predictor",
                f"{settings.BASE_DIR}/Barbershop/pretrained_models/shape_predictor_68_face_landmarks.dat",
            ]
            subprocess.run(align_cmd)
            storage.delete(
                os.path.join(settings.MEDIA_ROOT, "user_input", uploaded_file)
            )
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] ------ Finish detect and align face to center of image"
            )

        if not os.path.exists(
            os.path.join(settings.MEDIA_ROOT, "user_input", aligned_file)
        ):
            return Response(
                {"status": "Can not find a face in the input image"},
                status=status.HTTP_406_NOT_ACCEPTABLE,
            )

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] ------ Begin transfer hair style"
        )
        input_dir = f"{settings.MEDIA_ROOT}/user_input"
        template_dir = f"{settings.MEDIA_ROOT}/template"
        im_path1 = os.path.join(input_dir, aligned_file)
        im_path2 = os.path.join(template_dir, f"{target}.png")
        im_path3 = im_path2

        im_set = {im_path1, im_path2, im_path3}
        settings.MODEL.invert_images_in_W([*im_set])
        settings.MODEL.invert_images_in_FS([*im_set])

        align = Alignment(args)
        align.align_images(
            im_path1,
            im_path2,
            sign=args.sign,
            align_more_region=False,
            smooth=args.smooth,
        )

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ------ Step 3: Blend images")
        blend = Blending(args)
        blend.blend_images(im_path1, im_path2, im_path3, sign="realistic")

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] ------ Finish transfer hair style"
        )

        result_path = os.path.join(
            settings.MEDIA_ROOT,
            "user_output",
            f'{aligned_file.split(".")[0]}_result.png',
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ------ Received request")

        file_obj = request.FILES.get("file")
        target = request.POST.get("target")  # from 1 -> 8
        auto_crop = request.POST.get("auto_crop").lower() in ("yes", "true", "t", "1")
        print(f"AUTO_CROP: {auto_crop}")

        storage = FileSystemStorage()
        file_ext = file_obj.name.split(".")[-1]
        uploaded_file = f"{uuid.uuid1().hex}.{file_ext}"
        storage.save(
            os.path.join(settings.MEDIA_ROOT, "user_input", uploaded_file), file_obj
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ------ Stored original file")

        aligned_file = uploaded_file
        if auto_crop:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] ------ Begin detect and align face to center of image"
            )
            aligned_file = f"{uuid.uuid1().hex}.png"
            align_cmd = [
                "python",
                f"{settings.BASE_DIR}/Barbershop/align_face.py",
                "--input",
                f"{settings.MEDIA_ROOT}/user_input/{uploaded_file}",
                "--output",
                f"{settings.MEDIA_ROOT}/user_input/{aligned_file}",
                "--shape_predictor",
                f"{settings.BASE_DIR}/Barbershop/pretrained_models/shape_predictor_68_face_landmarks.dat",
            ]
            subprocess.run(align_cmd)
            storage.delete(
                os.path.join(settings.MEDIA_ROOT, "user_input", uploaded_file)
            )
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] ------ Finish detect and align face to center of image"
            )

        if not os.path.exists(
            os.path.join(settings.MEDIA_ROOT, "user_input", aligned_file)
        ):
            return Response(
                {"status": "Can not find a face in the input image"},
                status=status.HTTP_406_NOT_ACCEPTABLE,
            )

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] ------ Begin transfer hair style"
        )
        tranfer_cmd = [
            "python",
            f"{settings.BASE_DIR}/Barbershop/main.py",
            "--im_path1",
            aligned_file,
            "--im_path2",
            f"{target}.png",
            "--im_path3",
            f"{target}.png",
            "--learning_rate",
            "0.03",
            "--W_steps",
            "100",
            "--FS_steps",
            "100",
            "--align_steps1",
            "70",
            "--align_steps2",
            "50",
            "--blend_steps",
            "50",
            "--input_dir",
            f"{settings.MEDIA_ROOT}/user_input",
            "--output_dir",
            f"{settings.MEDIA_ROOT}/user_output",
            "--template_dir",
            f"{settings.MEDIA_ROOT}/template",
            "--ckpt",
            f"{settings.BASE_DIR}/Barbershop/pretrained_models/ffhq.pt",
            "--seg_ckpt",
            f"{settings.BASE_DIR}/Barbershop/pretrained_models/seg.pth",
        ]
        subprocess.run(tranfer_cmd)
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] ------ Finish transfer hair style"
        )

        result_path = os.path.join(
            settings.MEDIA_ROOT,
            "user_output",
            f'{aligned_file.split(".")[0]}_result.png',
        )
        if os.path.exists(result_path):
            return FileResponse(open(result_path, "rb"))
        else:
            return Response(
                {"status": "Hair transfer failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
