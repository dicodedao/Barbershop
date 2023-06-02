from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

from django.core.files.storage import FileSystemStorage
from django.core.files import File
from django.http import FileResponse

import barbershop_api.settings as settings
import uuid
import subprocess
import os

class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        target = request.POST.get('target') #from 1 -> 10
        
        storage = FileSystemStorage()
        file_ext = file_obj.name.split('.')[-1]
        uploaded_file = f'{uuid.uuid1().hex}.{file_ext}'
        storage.save(os.path.join(settings.MEDIA_ROOT, 'user_input', uploaded_file), file_obj)
        aligned_file = f'{uuid.uuid1().hex}.png'
        align_cmd = ['python', f'{settings.BASE_DIR}/Barbershop/align_face.py', '--input', f'{settings.MEDIA_ROOT}/user_input/{uploaded_file}', '--output', f'{settings.MEDIA_ROOT}/user_input/{aligned_file}', '--shape_predictor', f'{settings.BASE_DIR}/Barbershop/pretrained_models/shape_predictor_68_face_landmarks.dat']
        subprocess.run(align_cmd)
        storage.delete(os.path.join(settings.MEDIA_ROOT, 'user_input', uploaded_file))
        if not os.path.exists(os.path.join(settings.MEDIA_ROOT, 'user_input', aligned_file)):
            return Response({'status': 'Can not find a face in the input image'}, status=status.HTTP_406_NOT_ACCEPTABLE)
        tranfer_cmd = ['python', f'{settings.BASE_DIR}/Barbershop/main.py', '--im_path1', aligned_file, '--im_path2', f'{target}.png', '--im_path3', f'{target}.png', '--learning_rate', '0.03', '--W_steps', '100', '--FS_steps', '100', '--align_steps1', '70', '--align_steps2', '50', '--blend_steps', '50', '--input_dir', f'{settings.MEDIA_ROOT}/user_input', '--output_dir', f'{settings.MEDIA_ROOT}/user_output', '--template_dir', f'{settings.MEDIA_ROOT}/template', '--ckpt', f'{settings.BASE_DIR}/Barbershop/pretrained_models/ffhq.pt', '--seg_ckpt', f'{settings.BASE_DIR}/Barbershop/pretrained_models/seg.pth',]
        subprocess.run(tranfer_cmd)
        # storage.delete(aligned_file)
        result_path = os.path.join(settings.MEDIA_ROOT, 'user_output', f'{aligned_file.split(".")[0]}_result.png')
        if os.path.exists(result_path):
            return FileResponse(open(result_path, 'rb'))
        else:
            return Response({'status': 'Hair transfer failed'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
