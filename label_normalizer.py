import json
from pycocotools.coco import COCO
import re
import os
import cv2
from util import get_coco_empty_json

json_coco = get_coco_empty_json()

FILE_NAME_REGEX = r"([0-9]+\.jpg$)|(IMG_[0-9]+\.JPG$)"
file_name_regex = re.compile(FILE_NAME_REGEX, re.IGNORECASE)


DATASET_ROOT_DIR = "augmented_images/"
FILE_REGEX_AUGMENTATION = r"([0-9]+_[0-9]+\.jpg$)|(IMG_[0-9]+_[0-9]\.jpg$)"
file_aug_name_regex = re.compile(r"([0-9]+_[0-9]+\.jpg$)|(IMG_[0-9]+_[0-9]\.jpg$)", re.IGNORECASE)

def remove_augmented_images():
    for root, dirs, files in os.walk(DATASET_ROOT_DIR):
            print("remove augmented files ...")
            for file in files:
                if file_aug_name_regex.match(file):
                    os.remove(DATASET_ROOT_DIR+file)
    pass


def normalize_images_path(ann_file):
    imgs_count = 0
    labels_count = 0
    coco_file = COCO(ann_file)
    images_ids = list(coco_file.imgToAnns.keys())

    for image_id in images_ids:
        image_path = coco_file.imgs[image_id]['file_name']
        match = re.search(FILE_NAME_REGEX, image_path)
        
        if match:
            image_file_name = match.group()
            new_image_file_name = f"augmented_images/{image_file_name}"
            if os.path.isfile(new_image_file_name):
                bboxes = []
                labels = []
                image = cv2.imread(new_image_file_name)
                for ann in coco_file.imgToAnns[image_id]:
                    json_coco["annotations"].append({
                        "segmentation": ann["segmentation"],
                        "area": ann["area"],
                        "iscrowd": ann["iscrowd"],
                        "image_id": imgs_count,
                        "bbox": ann["bbox"],
                        "category_id": ann["category_id"],
                        "id": labels_count
                    })
                    labels_count += 1
            
                json_coco["images"].append({
                    "licence": 0,
                    "file_name": new_image_file_name,
                    "coco_url": "",
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "date_captured": "2019-11-01 00:00:00",
                    "flickr_url": "",
                    "id": imgs_count
                })
                imgs_count += 1
            else:
                print(f"image not found {new_image_file_name}")

        else:
            print(f"invalid image path {image_path} ")
    
    return json_coco
        

def normalize_labels():
    remove_augmented_images()
    # normalize_images_path("labels/Tomaszewski_CPLID_train.json")
    # normalize_images_path("labels/Tomaszewski_CPLID_test.json")

    # with open("labels/paid.json", "w") as output:
    #     json.dump(json_coco, output)


if __name__ == "__main__":
    normalize_labels()