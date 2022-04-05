import pandas as pd
import numpy as np

import mmcv
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os

image_path = '../data/train'
crop_image_path = '../data/train_crop'

CATEGORIES = ('cell',)
CAT2IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}

def init_coco():
    return {
        'info': {},
        'categories':
            [{
                'id': idx,
                'name': cat,
            } for cat, idx in CAT2IDX.items()]
    }

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def make_crop_coco(coco_path, save_path, mode='train'):
    coco = COCO(coco_path)
    img_infos = []
    ann_infos = []
    for ann in tqdm(coco.loadAnns(coco.getAnnIds())):
        x_min, y_min, width, height = ann['bbox']
        ann_id = ann['id']
        img = Image.open(os.path.join(image_path, coco.loadImgs(ann['image_id'])[0]['file_name']))
        
        # 이미지에서 bbox 부분 crop 후 저장
        crop_img = img.crop((x_min, y_min, x_min + width, y_min + height))
        crop_img.save(os.path.join(crop_image_path, mode+str(ann_id)+'.png'))
        
        img_info = dict(
                id=ann_id,
                width=width,
                height=height,
                file_name=mode+str(ann_id)+'.png',
            )
        
        mask_np = mask_utils.decode(ann['segmentation'])
        crop_mask = mask_np[int(y_min):int(y_min+height), int(x_min):int(x_min+width)]
        rle = mask_utils.encode(np.asfortranarray(crop_mask))
        rle['counts'] = rle['counts'].decode()
        bbox = mask_utils.toBbox(rle).tolist()

        ann_info = dict(
                    id=ann_id,
                    image_id=ann_id,
                    category_id=0,
                    iscrowd=0,
                    segmentation=rle,
                    area=bbox[2] * bbox[3],
                    bbox=bbox,
                )
        img_infos.append(img_info)
        ann_infos.append(ann_info)

        # if ann['id'] == 5:
        #     break

    save_coco = init_coco()
    save_coco['images'] = img_infos
    save_coco['annotations'] = ann_infos
    mmcv.dump(save_coco, save_path)


def main():
    make_dir(crop_image_path)
    make_crop_coco('../data/dtrain_g0.json', '../data/dtrain_g0_crop.json', 'train')
    make_crop_coco('../data/dval_g0.json', '../data/dval_g0_crop.json', 'valid')


if __name__ == '__main__':
    main()