import argparse
import numpy as np
from skimage.segmentation import felzenszwalb
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.deeplabv3_version_1_256.deeplabv3 import DeepLabV3
from models.U_net.unet_model import UNet
from models.FPN.FPN import FPN
from models.FCN.fcn import FCNs, FCN8s, FCN16s, FCN32s, VGGNet
from models.Segnet.Segnet import SegNet
from models.GCN.FCN_GCN import FCN_GCN
from models.PSPNEt.pspnet import PSPNet
from models.deepLabv3plus.deeplabv3plus import deeplabv3plus
from models.deepLabv3plus.config import cfg
from torch.autograd import Variable
import torch
import os
import pandas as pd
from PIL import Image
import cv2
from collections import OrderedDict
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from palette import colorize_mask

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def snapshot_forward(model, dataloader, model_list, png, save_path):
    model.eval()
    for (index, (image, pos_list, filename)) in enumerate(dataloader):
        image = Variable(image).cuda()
        # print(building_image)
        # print(pos_list)

        predict_list = 0
        for model in model_list:
            predict_1 = model(image)
            # predict_list = predict_1
            predict_2 = model(torch.flip(image, [-1]))
            predict_2 = torch.flip(predict_2, [-1])

            predict_3 = model(torch.flip(image, [-2]))
            predict_3 = torch.flip(predict_3, [-2])

            predict_4 = model(torch.flip(image, [-1, -2]))
            predict_4 = torch.flip(predict_4, [-1, -2])

            predict_list += (predict_1 + predict_2 + predict_3 + predict_4)
        predict_list = torch.argmax(predict_list.cpu(), 1).byte().numpy()  # n x h x w

        batch_size = predict_list.shape[0]  # batch大小
        for i in range(batch_size):
            predict = predict_list[i]
            pos = pos_list[i, :]
            [topleft_x, topleft_y, buttomright_x, buttomright_y] = pos

            if (buttomright_x - topleft_x) == 512 and (buttomright_y - topleft_y) == 512:
                png[topleft_y + 128:buttomright_y - 128, topleft_x + 128:buttomright_x - 128] = predict[128:384,
                                                                                                128:384]
                save = save_path + "\\image_supervised_no_pre_DeepLabv3plus"
                if not os.path.exists(save):
                    os.makedirs(save)
                overlap = colorize_mask(predict)
                label_save_path = save + '\\' + filename[i][:-4] + '_predict.tif'
                overlap.save(label_save_path)
                # cv2.imwrite(save + '\\' +
                #             filename[i][:-4] + '_predict.tif', predict)
            else:
                raise ValueError(
                    "target_size!=512， Got {},{}".format(buttomright_x - topleft_x, buttomright_y - topleft_y))
    h, w = png.shape
    png = png[128:h - 128, 128:w - 128]
    zero = (6800, 7200)  # 去除补全512整数倍时的右下边界
    # zero = (11022,10983)  # 去除补全512整数倍时的右下边界
    png = png[:zero[0], :zero[1]]
    return png


def parse_args():
    parser = argparse.ArgumentParser(description="膨胀预测")
    parser.add_argument('--test-data-root', type=str,
                        default=r'C:\code\dxj\High-Resolution-Remote-Sensing-Semantic-Segmentation-PyTorch-master\tools\data_512\complete_image')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='batch size for testing (default:16)')
    parser.add_argument('--num_workers', type=int, default=2)
    # #Deeplab
    parser.add_argument("--model-path", type=str,
                        # default=r"D:\dataset\Graduation_project\model\512\U_net_final\resnet50\background\epoch_25_acc_0.77646_kappa_0.68166.pth"
                        # default=r"D:\dataset\Graduation_project\model\512\PSPNet\resnet50\background\epoch_15_acc_0.77735_kappa_0.68602.pth"
                        # default=r"D:\dataset\Graduation_project\model\512\GCN\resnet50\background\epoch_17_acc_0.78079_kappa_0.69210.pth"
                        default=r"D:\dataset\Graduation_project\model\512\Deeplabv3Plus\resnet50\background\epoch_16_acc_0.78867_kappa_0.70419.pth"
                        # default=r"D:\dataset\Graduation_project\model\512\FCN8s\resnet50\background\epoch_12_acc_0.77841_kappa_0.68538.pth"
                        # default=r"D:\dataset\Graduation_project\model\512\FCN32s\resnet50\background\epoch_10_acc_0.77525_kappa_0.68374.pth"
                        )
    # parser.add_argument("--pred-path", type=str, default="")
    # U_net
    # parser.add_argument("--model-path", type=str,
    #                     default=r"D:\dataset\Graduation_project\model\512\U_net_final\resnet50\background\\epoch_25_acc_0.77646_kappa_0.68166.pth")
    parser.add_argument("--pred-path", type=str, default="")
    args = parser.parse_args()

    return args


def create_png():
    zeros = (6800, 7200)
    # zeros = (11022,10983)
    h, w = zeros[0], zeros[1]
    new_h, new_w = (h // 512 + 1) * 512, (w // 512 + 1) * 512  # 填充下边界和右边界得到滑窗的整数倍
    zeros = (new_h + 128, new_w + 128)  # 填充空白边界，考虑到边缘数据
    zeros = np.zeros(zeros, np.uint8)
    return zeros


class Inference_Dataset(Dataset):
    def __init__(self, root_dir, csv_file, transforms):
        self.root_dir = root_dir
        self.csv_file = pd.read_csv(csv_file, header=None)
        self.transforms = transforms

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        filename = self.csv_file.iloc[idx, 0]
        # print(filename)
        image_path = os.path.join(self.root_dir, filename)
        # building_image = np.asarray(Image.open(image_path))  # mode:RGBA
        # building_image = cv.cvtColor(building_image, cv.COLOR_RGBA2BGRA)  # PIL(RGBA)-->cv2(BGRA)
        image = Image.open(image_path).convert('RGB')

        # if self.transforms:
        #     print('transforms')
        image = self.transforms(image)

        pos_list = self.csv_file.iloc[idx, 1:].values.astype(
            "int")  # ---> (topleft_x,topleft_y,buttomright_x,buttomright_y)
        return image, pos_list, filename


def reference(csv_file, save_path, image_path):
    args = parse_args()
    dataset = Inference_Dataset(root_dir=image_path,
                                csv_file=csv_file,
                                transforms=img_transform)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4)
    n_blocks = '3, 4, 23, 3'
    n_blocks = [int(b) for b in n_blocks.split(',')]
    # model = FPN(num_classes=6, num_blocks=n_blocks)
    # model = UNet(n_channels=3, n_classes=6)
    # model = DeepLabV3(num_classes=6)
    # model = SegNet(3, 6)
    # model = PSPNet(6, 8)
    # model = FCN_GCN(6)
    model = deeplabv3plus(cfg)
    # vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    # model = FCNs(pretrained_net=vgg_model, n_class=6)
    state_dict = torch.load(args.model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model_list = []
    model_list.append(model)
    zeros = create_png()
    image = snapshot_forward(model, dataloader, model_list, zeros, save_path)
    label_save_path = save_path + "\\vis_predict_image_DeepLabv3plus.png"

    overlap = colorize_mask(image)
    # overlap.show()
    overlap.save(label_save_path)


if __name__ == '__main__':
    csv_file_list = [ ]

    save_path = [ ]

    image_path = [ ]
    for i in tqdm(range(len(csv_file_list))):
    # for i in tqdm(range(0, 15)):
        reference(csv_file_list[i], save_path[i], image_path[i])
