import numpy as np
import argparse
import numpy as np
import cv2
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import gdal
import re


# zeros = (10782, 9454)
# zeros = (11022,10983)
def image_merge(csv_file, root_dir, save_dir):
    root_dir = root_dir
    csv_file = pd.read_csv(csv_file, header=None)

    for idx in tqdm(range(csv_file.shape[0])):
        image_list = []
        filename = csv_file.iloc[idx, 0]
        # print(filename)
        # image_path = os.path.join(file.path, filename[:-4]+'_segemntation.tif')
        image_root_list = os.listdir(root_dir)
        image_path_part = filename[:-4] + '_predict'
        # image_path_part = filename[:28] + '_crf_' + str(idx) + '.tif'
        for path in image_root_list:
            if image_path_part in path:
                image_path = path
                break
        image_path = os.path.join(root_dir, image_path)

        dataset = gdal.Open(image_path)
        label = dataset.GetRasterBand(1).ReadAsArray()
        pos_list = csv_file.iloc[idx, 1:].values.astype("int")
        [topleft_x, topleft_y, buttomright_x, buttomright_y] = pos_list
        if (buttomright_x - topleft_x) == 512 and (buttomright_y - topleft_y) == 512:
            zeros[topleft_y + 128:buttomright_y - 128, topleft_x + 128:buttomright_x - 128] = label[128:384, 128:384]

    h, w = zeros.shape
    png = zeros[128:h - 128, 128:w - 128]
    # zero = (10782, 9454)
    zero = (6800, 7200)
    # zero = (11022, 10983)
    png = png[:zero[0], :zero[1]]

    label_save_path = os.path.join(save_dir, 'image_only_pre_GCN') + '.png'
    print(label_save_path)
    from palette import colorize_mask

    overlap = colorize_mask(png)
    overlap.save(label_save_path)


if __name__ == "__main__":
    zeros = (6800, 7200)
    h, w = zeros[0], zeros[1]
    new_h, new_w = (h // 512 + 1) * 512, (w // 512 + 1) * 512  # 填充下边界和右边界得到滑窗的整数倍
    zeros = (new_h + 128, new_w + 128)  # 填充空白边界，考虑到边缘数据
    zeros = np.zeros(zeros, np.uint8)
    csv_file_list = []
    root_dir_list = []
    save_dir_list = []
    for k in range(17):
        root_dir_list.append(r"D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_" + str(
            k + 1) + "\image_supervised_GCN")
        save_dir_list.append(r"D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_" + str(
            k + 1))
    for i in range(17):
        image_merge(csv_file_list[i], root_dir_list[i], save_dir=save_dir_list[i])
