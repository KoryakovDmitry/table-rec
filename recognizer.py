import sys

sys.path.append('/home/ochinchin/Initflow/OcrRPA/cascade20/table-rec/mmdetection')

from border import border
from mmdet.apis import inference_detector, init_detector
import cv2
from blessfunc import borderless
import glob

import numpy as np
import pandas as pd

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

image_path = '/home/ochinchin/Initflow/OcrRPA/cascade20/table-rec/img/*'
config_fname = '/home/ochinchin/Initflow/OcrRPA/cascade20/table-rec/cascade_mask_rcnn_hrnetv2p_w32_20e.py'
checkpoint_path = "/home/ochinchin/Initflow/OcrRPA/cascade20/table-rec/epoch_36_mmdet20_converted.pth"
device = 'cuda:0'
threshold = .55

import matplotlib.pyplot as plt

def ocr_core(filename, coord_to_crop):
    """
    This function will handle the core OCR processing of images.
    """
    text = pytesseract.image_to_string(Image.open(filename).crop(coord_to_crop))
    return text


def table_rec(image_path=image_path, config_fname=config_fname, checkpoint_path=checkpoint_path,
         device=device, threshold=threshold):
    global tables
    model = init_detector(config=config_fname, checkpoint=checkpoint_path, device=device)

    # List of images in the image_path
    imgs = glob.glob(image_path)
    tables = []
    for i in imgs:
        result = inference_detector(model, i)
        res_border = []
        res_bless = []
        res_cell = []
        ## for border
        for r in result[0][0]:
            if r[4] > threshold:
                res_border.append(r[:4].astype(int))
        ## for cells
        for r in result[0][1]:
            if r[4] > threshold:
                r[4] = r[4] * 100
                res_cell.append(r.astype(int))
        ## for borderless
        for r in result[0][2]:
            if r[4] > threshold:
                res_bless.append(r[:4].astype(int))

        ## if border tables detected
        if len(res_border) != 0:
            ## call border script for each table in image
            for res in res_border:
                try:
                    tables.append(border(res, cv2.imread(i)))
                except:
                    pass
        if len(res_bless) != 0:
            if len(res_cell) != 0:
                for no, res in enumerate(res_bless):
                    tables.append(borderless(res, cv2.imread(i), res_cell))
    n_col = np.array(tables[0][1]).max(0)[0, 0]
    n_row = np.array(tables[0][1]).max(0)[1, 0]
    df = pd.DataFrame(np.nan, index=[_ for _ in range(n_row)], columns=[_ for _ in range(n_col)])
    for table in tables:
        for cell in table[1]:
            print(cell[0][1] + cell[1][1])
            df.iloc[cell[0][0], cell[1][0]] = ocr_core(imgs[0], cell[0][1] + cell[1][1])
    print(df)
    return tables


if __name__ == '__main__':
    table_rec()
