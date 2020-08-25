import cv2
import os
import pytesseract
from border import border
from mmdet.apis import inference_detector, init_detector
from blessfunc import borderless

import numpy as np
import pandas as pd
import glob

try:
    from PIL import Image
except ImportError:
    import Image

config_fname = 'cascade_mask_rcnn_hrnetv2p_w32_20e.py'
checkpoint_path = "epoch_36_mmdet_2.0_converted.pth"
device = 'cpu'
threshold = 1e-6


def ocr_core(filename, coord_to_crop):
    """
    This function will handle the core OCR processing of images.
    """
    # Image.open(filename).crop(
    #     (coord_to_crop[0] - 5, coord_to_crop[1] - 5, coord_to_crop[2] + 5, coord_to_crop[3] + 5)).show()
    text = pytesseract.image_to_string(Image.open(filename).crop((coord_to_crop[0] - 5,
                                                                  coord_to_crop[1] - 5,
                                                                  coord_to_crop[2] + 5,
                                                                  coord_to_crop[3] + 5)))
    return text


def show_cells(res_cell, image_path):
    for cell in res_cell:
        Image.open(image_path).crop((cell[0] - 5, cell[1] - 5, cell[2] + 5, cell[3] + 5)).show()


def table_rec(image_path, config_fname=config_fname, checkpoint_path=checkpoint_path,
              device=device, threshold=threshold, need_show_cells=False):
    global tables
    model = init_detector(config=config_fname, checkpoint=checkpoint_path, device=device)

    # List of images in the image_path
    tables = []
    result = inference_detector(model, image_path)
    res_border = []
    res_bless = []
    res_cell = []
    # for border
    for r in result[0][0]:
        if r[4] > threshold:
            res_border.append(r[:4].astype(int))
    # for cells
    for r in result[0][1]:
        if r[4] > threshold:
            r[4] = r[4] * 100
            res_cell.append(r.astype(int))
    # for borderless
    for r in result[0][2]:
        if r[4] > threshold:
            res_bless.append(r[:4].astype(int))

    # if border tables detected
    if len(res_border) != 0:
        # call border script for each table in image
        for res in res_border:
            try:
                tables.append(border(res, cv2.imread(image_path)))
            except:
                pass
    if len(res_bless) != 0:
        if len(res_cell) != 0:
            for res in res_bless:
                tables.append(borderless(res, cv2.imread(image_path), res_cell))
    if need_show_cells:
        show_cells(res_cell, image_path)
    if (len(res_bless) != 0) or (len(res_border) != 0):
        for num, table in enumerate(tables):
            n_col = np.array(tables[num][1]).max(0)[0, 0]
            n_row = np.array(tables[num][1]).max(0)[1, 0]
            df = pd.DataFrame(np.nan, index=[_ for _ in range(n_row + 1)], columns=[_ for _ in range(n_col + 1)])
            for cell in table[1]:
                # print(cell[0][1] + cell[1][1])
                df.iloc[cell[1][0], cell[0][0]] = ocr_core(image_path, cell[0][1] + cell[1][1]).rstrip()
                # print(df)
            new_header = df.iloc[0]  # grab the first row for the header
            df = df[1:]  # take the data less the header row
            df.columns = new_header
            df.dropna(how='all')
            df.to_csv(f'table_{os.path.splitext(os.path.basename(image_path))[0]}_{num}.csv')
    else:
        print(f'tables not found {os.path.splitext(os.path.basename(image_path))[0]}')


if __name__ == '__main__':
    # table_rec(image_path='/Users/dmitry/Documents/Initflow/invoice-recognition/table-rec/imgs/NonTaxInvoice.png')
    for img in glob.glob('imgs/*'):
        table_rec(image_path=img)
