import numpy as np
import cv2
from libs import average_meter, metric
from class_names import five_classes
from prettytable import PrettyTable
from osgeo import gdal, gdalconst

label_path = [r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS1__L1A0001670888-MSS1_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS1__L1A0001821711-MSS1_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS1__L1A0001910522-MSS1_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS2__L1A0000635115-MSS2_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS2__L1A0001092725-MSS2_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS2__L1A0001119057-MSS2_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS2__L1A0001246644-MSS2_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS2__L1A0001396036-MSS2_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS2__L1A0001787080-MSS2_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS2__L1A0000564692-MSS2_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS1__L1A0001680851-MSS1_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS2__L1A0001116444-MSS2_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS2__L1A0000607681-MSS2_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS2__L1A0001787089-MSS2_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS1__L1A0001680853-MSS1_label.tif',
              r'D:\dataset\Graduation_project\Graduation_result\512\image_val\GF2_PMS1__L1A0001734328-MSS1_label.tif',

              ]

predict_path = [
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\vis_predict_image_U_net.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\vis_predict_image_U_net_crf.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\image_unsupervised_16\image_0.6_U_net_no_pre.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\image_unsupervised_16\image_0.6_U_net.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\vis_predict_image.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\vis_predict_image_GCN.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\vis_predict_image_SegNet.png',
    r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\vis_predict_image_GCN.png',
    r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\image_only_pre_GCN.png',
    r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\image_unsupervised_16\image_0.6_GCN.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\vis_predict_image_DeepLabv3plus.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\vis_predict_image.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\image_unsupervised_16\image_0.6.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\image_unsupervised_16\image_0.5_no_pre.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\vis_predict_image_FCNs.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\vis_predict_image_FCN8s.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\vis_predict_image_FCN16s.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\vis_predict_image_FCN32s.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\image_unsupervised_16\image_0.6_no_pre_FCN32s.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\image_unsupervised_16\image_0.6_no_pre_FCNs.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\image_unsupervised_16\image_0.7_no_pre.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\image_unsupervised_16\image_0.8_no_pre.png',
    # r'D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_4\image_only_pre_U_net.png',
    ]

acc = []

image_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


for j in range(len(predict_path)):
    for k in image_number:
        mask = cv2.imread(label_path[k - 1], 0)
        path = predict_path[j].replace('_val_4', '_val_' + str(k)).replace('only_pre', 'only_pre')
        print(path)
        referset = gdal.Open(path, gdalconst.GA_ReadOnly)

        preds = referset.GetRasterBand(1).ReadAsArray()
        mask[(mask != 76) & (mask != 150) & (mask != 179) & (mask != 226) & (mask != 29)] = 0
        mask[mask == 76] = 1
        mask[mask == 150] = 2
        mask[mask == 179] = 3
        mask[mask == 226] = 4
        mask[mask == 29] = 5

        if k == image_number[0]:
            preds_all = preds.flatten()
            label_all = mask.flatten()
        else:
            label_all = np.hstack((label_all, mask.flatten()))
            preds_all = np.hstack((preds_all, preds.flatten()))

    class_names = five_classes()

    num_classes = 6

    # conf_mat = metric.confusion_matrix_1(pred=preds.flatten(),
    #                                      label=mask.flatten(),
    #                                      num_classes=num_classes)
    conf_mat = metric.confusion_matrix_1(pred=preds_all,
                                         label=label_all,
                                         num_classes=num_classes)

    table = PrettyTable(["序号", "名称", "acc", "IoU"])
    train_acc, train_acc_per_class, train_acc_cls, train_IoU, train_mean_IoU, train_kappa = metric.evaluate(
        conf_mat)

    for i in range(num_classes):
        table.add_row([i, class_names[i], train_acc_per_class[i], train_IoU[i]])
    print(table)
    print("train_acc:", train_acc)
    print("train_mean_IoU:", train_mean_IoU)
    print("kappa:", train_kappa)
    acc.append(train_acc)

print(acc)
