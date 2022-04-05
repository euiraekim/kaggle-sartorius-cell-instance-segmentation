import numpy as np
from PIL import Image
import os
from tqdm import tqdm

import mmcv
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils

image_path = '../data/train'
classes = ('cell', 'bg')
palette = [[0, 0, 0], [255, 255, 255]]

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def convert_to_mmseg(coco_path, save_path, mode='train'):
    image_save_path = os.path.join(save_path, 'images', mode)
    ann_save_path = os.path.join(save_path, 'annotations', mode)
    make_dir(image_save_path)
    make_dir(ann_save_path)

    coco = COCO(coco_path)
    for ann in tqdm(coco.loadAnns(coco.getAnnIds())):
        x_min, y_min, width, height = ann['bbox']
        ann_id = ann['id']
        img = Image.open(os.path.join(image_path, coco.loadImgs(ann['image_id'])[0]['file_name']))
        
        # 이미지에서 bbox 부분 crop 후 저장
        crop_img = img.crop((x_min, y_min, x_min + width, y_min + height))
        crop_img.save(os.path.join(image_save_path, str(ann_id)+'.png'))
        
        mask_np = mask_utils.decode(ann['segmentation'])
        crop_mask = mask_np[int(y_min):int(y_min+height), int(x_min):int(x_min+width)]
        seg_img = Image.fromarray(crop_mask).convert('P')
        seg_img.putpalette(np.array(palette, dtype=np.uint8))
        seg_img.save(os.path.join(ann_save_path, str(ann_id)+'.png'))

        # if ann['id'] == 5:
        #     break


def main():
    convert_to_mmseg('../data/dtrain_g0.json', '../data/segmentation', 'train_g0')
    convert_to_mmseg('../data/dval_g0.json', '../data/segmentation', 'valid_g0')


if __name__ == '__main__':
    main()