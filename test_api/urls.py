from django.urls import path
from .views import FileUploadView2

urlpatterns = [
    path("upload", FileUploadView2.as_view()),
]
