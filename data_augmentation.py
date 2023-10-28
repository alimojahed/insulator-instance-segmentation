import os
import errno
import random
import json
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from albumentations import (
    BboxParams,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Cutout,
    CoarseDropout,
    Crop,
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)

from util import get_coco_empty_json, create_sub_mask_annotation
print("test")
from segment_anything import sam_model_registry, SamPredictor
print("test")
import torch
print("test")

sam_checkpoint = "models/sam_vit_l_0b3195.pth"
model_type = "vit_l"


    

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_built() else "cpu")
print("test0")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
print("test1")
sam.to(device=device)
print("test2")
predictor = SamPredictor(sam)
print("test3")
def get_segmentation_mask(bbox):
    return predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox[None, :],
        multimask_output=False,
    )

print("test")
def strong_aug(img_shape, p=0.5):
    return [
        RandomRotate90(),
        Flip(),
        CoarseDropout(2,int(img_shape[0]*0.1),int(img_shape[1]*0.1),1,int(img_shape[0]*0.05),int(img_shape[1]*0.05),p=1.), # mudar para 5 a 10 % do tamanho da imagem
        OneOf([
            IAAAdditiveGaussianNoise(scale=(0.05 * 255, 0.1 * 255)), # default: (0.01 * 255, 0.05 * 255)
            GaussNoise(),
        ], p=1.),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=1.),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=(0.0,0.3), rotate_limit=25, p=0.35),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            RandomBrightnessContrast(brightness_limit=0.3), #default: 0.2
        ], p=1.),
    ]

def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params=BboxParams(format='coco', min_area=min_area, 
                                               min_visibility=min_visibility, label_fields=['category_id']))


################## PARAMETERS ##################
balancing_factor = 10
training_ratio = 0.2
data_root = 'augmented_images/'
################################################
print("test")
json_coco = get_coco_empty_json()

def augment(labels):
    
    imgs_count = 0
    labels_count = 0
    print("test")
    coco_file = COCO(labels)
    
    img_ids = list(coco_file.imgToAnns.keys())

    print(img_ids)
    for index in tqdm(range(len(img_ids))):
        img_id = img_ids[index]
        img_path = coco_file.imgs[img_id]['file_name']
        image = cv2.imread(data_root + img_path)
        bboxes = []
        labels = []
        
        for ann in coco_file.imgToAnns[img_id]:
            mask = get_segmentation_mask(ann["bbox"])
            segmentations = create_sub_mask_annotation(mask)
            print(segmentations)
            json_coco["annotations"].append({
                "segmentation": segmentations,
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
            "file_name": img_path,
            "coco_url": "",
            "height": image.shape[0],
            "width": image.shape[1],
            "date_captured": "2019-11-01 00:00:00",
            "flickr_url": "",
            "id": imgs_count
        })
        imgs_count += 1

        # cv2.imwrite(data_root + os.path.basename(img_path), image)

        for ann in coco_file.imgToAnns[img_id]:
            bboxes.append(ann['bbox'])
            labels.append(ann['category_id'])

        annotation = {'image': image, 'bboxes': bboxes, 'category_id': labels}

        for i in range(balancing_factor-1):
            augmentation = get_aug(strong_aug(image.shape, p=1,))
            augmented = augmentation(**annotation)

            new_image_path = data_root + os.path.basename(img_path).replace('.jpg', '_' + str(i+1) + '.jpg').replace('.JPG', '_' + str(i+1) + '.jpg')
            cv2.imwrite(new_image_path, augmented['image'])

            for index, ann in enumerate(augmented['bboxes']):
                mask = get_segmentation_mask(ann)
                segmentations = create_sub_mask_annotation(mask)
                json_coco["annotations"].append({
                    "segmentation": [],
                    "area": ann[2] * ann[3],
                    "iscrowd": 0,
                    "image_id": imgs_count,
                    "bbox": ann,
                    "category_id": augmented['category_id'][index],
                    "id": labels_count
                })
                labels_count += 1
                
            json_coco["images"].append({
                "licence": 0,
                "file_name": new_image_path,
                "coco_url": "",
                "height": image.shape[0],
                "width": image.shape[1],
                "date_captured": "2019-11-01 00:00:00",
                "flickr_url": "",
                "id": imgs_count
            })
            imgs_count += 1
    output = open("labels/paid_aug" + ".json", "w")
    json.dump(json_coco, output)
    output.close()
    print("FINISHED")

if __name__ == "__main__":
    augment("labels/paid.json")