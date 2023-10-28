import wget
import os

if not os.path.exists("models"):
    os.mkdir("models")

sam_model = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
print("downloading SAM (Segment Anything Model) weights")
wget.download(sam_model, out="models")
