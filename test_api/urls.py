from django.urls import path
from .views import *

urlpatterns = [
    path("upload", FileUploadView.as_view()),
    path("transfer", TransferView.as_view()),
    path("template", CreateTemplateView.as_view()),
]
