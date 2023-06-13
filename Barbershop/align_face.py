import dlib
from pathlib import Path
import argparse
import torchvision
from Barbershop.utils.drive import open_url
from Barbershop.utils.shape_predictor import align_face
import PIL

parser = argparse.ArgumentParser(description="Align_face")

parser.add_argument("--input", type=str, help="input file")
parser.add_argument("--output", type=str, help="output file")
parser.add_argument(
    "--output_size",
    type=int,
    default=1024,
    help="size to downscale the input images to, must be power of 2",
)
parser.add_argument(
    "--shape_predictor",
    type=str,
    default="pretrained_models/shape_predictor_68_face_landmarks.dat",
    help="Shape predictor model",
)
args = parser.parse_args()
# f=open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=cache_dir, return_path=True)
predictor = dlib.shape_predictor(args.shape_predictor)

faces = align_face(str(args.input), predictor)
if len(faces) == 1:
    face = faces[0]
    if args.output_size:
        factor = 1024 // args.output_size
        assert args.output_size * factor == 1024
        face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
        face_tensor_lr = face_tensor[0].cpu().detach().clamp(0, 1)
        face = torchvision.transforms.ToPILImage()(face_tensor_lr)
        if factor != 1:
            face = face.resize((args.output_size, args.output_size), PIL.Image.LANCZOS)
    face.save(args.output)
