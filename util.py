import json
from skimage import measure                  
from shapely.geometry import Polygon, MultiPolygon
import numpy as np

def get_coco_empty_json():
    json_coco = json.loads("{}")
    json_coco["info"] = {
            "description": "PAID dataset: a Public Augmented Insulator Defect dataset",
            "url": "",
            "version": "1.0",
            "year": 2020,
            "contributor": "Voxar Labs - Centro de Informática - Universidade Federal de Pernambuco - Brazil",
            "date_created": "2020/01/17"
    }
    json_coco["licenses"] = [
        {
            "url": "https://opensource.org/licenses/MIT",
            "id": 0,
            "name": "MIT"
        }
    ]
    json_coco["categories"] = [
        {
            "supercategory": "insulator",
            "id": 0,
            "name": "normal_insulator"
        },
        {
            "supercategory": "insulator",
            "id": 1,
            "name": "defective_insulators"
        },
        {
            "supercategory": "insulator",
            "id": 2,
            "name": "insulators_fault"
        }
    ]

    json_coco["images"] = []
    json_coco["annotations"] = [] 

    return json_coco


def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)=

    contours = measure.find_contours(sub_mask[2][0], 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        if len(contour[0]) < 4:
            continue
        poly = Polygon(contour)
        
        poly = poly.simplify(1.0, preserve_topology=False)
        if isinstance(poly, MultiPolygon):
            continue
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)
        
    
    

    return  segmentations