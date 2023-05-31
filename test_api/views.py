from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response

from django.core.files.storage import FileSystemStorage
from django.core.files import File

import barbershop_api.settings as settings
import uuid
import subprocess

class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        target = request.POST.get('target') #from 1 -> 10
        
        storage = FileSystemStorage()
        file_ext = file_obj.name.split('.')[-1]
        file_name = storage.save(uuid.uuid1().hex + "." + file_ext, file_obj)


        command = ['python', f'{settings.BASE_DIR}/Barbershop/main.py', '--im_path1', file_name, '--img_path2', f'{target}.png', '--img_path3', f'{target}.png', '--learning_rate', 0.02, '--W_steps', 100, '--FS_steps', 100, '--align_steps1', 70, '--align_steps2', 50, '--blend_steps', 100, '--input_dir', '../media', '--output_dir', '../media']
        process = subprocess.Popen(command)
        process.wait()
        
        # Do something with the file_obj and text_field
        return Response({'status': 'success'})