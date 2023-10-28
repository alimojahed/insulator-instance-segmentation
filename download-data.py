import gdown
import os.path as path
from zipfile import ZipFile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--unzip', action="store_true")

args = parser.parse_args()

AUGMENTED_DATASET_FILE_PATH = "./insulators_datasets_merged.zip"

PAID_AUGMENTED_DATASET = "1TX7lfGfUcpLPmsir4DA57vPL6Y1BEOxz"

FORCE_UNZIP = args.unzip

if not path.isfile(AUGMENTED_DATASET_FILE_PATH):
    print("no dataset is available downloading from drive ...")
    
    gdown.download(id=PAID_AUGMENTED_DATASET, quiet=True)
    
    print("dataset downloaded")
    FORCE_UNZIP = True
else:
    print("dataset already downloaded")

if FORCE_UNZIP:
    print("unziping dataset")

    with ZipFile(AUGMENTED_DATASET_FILE_PATH, "r") as zipfile:
        zipfile.extractall()
    

    