import argparse
import numpy as np
from skimage.segmentation import felzenszwalb
from torch.autograd import Variable
from models.deeplabv3_version_1_256.deeplabv3 import DeepLabV3
from models.U_net.unet_model import UNet
from models.FPN.FPN import FPN
from models.Segnet.Segnet import SegNet
from models.GCN.FCN_GCN import FCN_GCN
from models.PSPNEt.pspnet import PSPNet
from models.deepLabv3plus.deeplabv3plus import deeplabv3plus
import torch
import os
import pandas as pd
from PIL import Image
import cv2
from collections import OrderedDict
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from collections import Counter
import torchvision.models as models
import torch.nn.functional as F
import torchvision
from multiprocessing import Pool
from palette import colorize_mask
import re
import gdal

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
        )

    def forward(self, x):
        return self.seq(x)


def segmentation_ML(image, scale, sigma, min_size):
    seg_map = felzenszwalb(image.transpose((0, 1, 2)), scale=scale, sigma=sigma, min_size=min_size)
    # seg_map = segmentation.slic(building_image.transpose((1, 2, 0)), n_segments=100000, compactness=100)
    # plt.imshow(seg_map)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]
    return seg_lab


def train_init(image, lr, momentum, dim):
    args_train = Args_train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    tensor = image.transpose((2, 0, 1))
    #     tensor = building_image
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    #     tensor = tensor[np.newaxis]
    tensor = torch.from_numpy(tensor).cuda()

    model = MyNet(inp_dim=dim, mod_dim1=args_train.mod_dim1, mod_dim2=args_train.mod_dim2)
    # model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return tensor, model, criterion, optimizer, device


class Args_train(object):
    train_epoch = 25
    mod_dim1 = 64  #
    mod_dim2 = 64

    min_label_num = 8  # if the label number small than it, break loop
    max_label_num = 256  # if the label number small than it, start to show result building_image.


def train(epoch, tensor, seg_lab, model, optimizer, criterion, image, min_label_num, path, save_path_trans):
    save_path_part = save_path_trans
    if not os.path.exists(save_path_part):
        os.makedirs(save_path_part)
    model.train()
    args_train = Args_train()
    for batch_idx in range(args_train.train_epoch):
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0]
        output = output.permute(1, 2, 0).view(-1, args_train.mod_dim2)
        target = torch.argmax(output, 1)

        im_target = target.data.cpu().numpy()
        '''refine'''
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]
        '''backward'''
        target = torch.from_numpy(im_target)
        target = target.cuda()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        '''show building_image'''
        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        #     print(un_label)
        if un_label.shape[0] < args_train.max_label_num:  # update show
            show = im_target.reshape(image[:, :, 0].shape)
        if len(un_label) <= min_label_num:
            break
    #         print('Loss:', batch_idx, loss.item())
    if not os.path.exists(save_path_part):
        os.makedirs(save_path_part)
    overlap = colorize_mask(show)
    label_save_path = save_path_part + '\\' + path.split('\\')[-1][:-4] + '_cnn.tif'
    overlap.save(label_save_path)

    cv2.imwrite('D:\\dataset\\Graduation_project\\Graduation_result\\predict\\cnn' + path.split('\\')[-1], show)
    return show


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def snapshot_forward(dataloader):
    # print(save_unsupervised_path,dataloader[4])
    save_path_part = save_unsupervised_path[dataloader[4]]
    label_list = [0, 0, 0, 0, 0, 0]
    args_train = Args_train()
    sigma = 0.8
    lr = 0.03
    min_label_num = 16
    min_size = 64
    scale = 64
    dim_num = 3

    image = dataloader[0].unsqueeze(0)
    pos_list = dataloader[1]
    image_path = dataloader[2]
    image_classify = dataloader[3].unsqueeze(0)
    image_mask = cv2.imread(image_path)

    seg_lab = segmentation_ML(image_mask, scale, sigma, min_size)
    seg_lab = np.array(seg_lab)
    if not os.path.exists(save_csv_path[dataloader[4]]):
        os.makedirs(save_csv_path[dataloader[4]])
    np.save(save_csv_path[dataloader[4]] + '\\' + image_path.split('\\')[-1][:-4] + '_seg.npy', seg_lab)

    tensor, model, criterion, optimizer, device = train_init(image_mask, lr, 0.9, dim_num)

    show = train(args_train.train_epoch, tensor, seg_lab, model, optimizer, criterion, image_mask, min_label_num,
                 image_path, save_path_part)

    # first_unsupervised
    image = Variable(image).cuda()
    predict_list = 0

    image_classify = image_classify.cuda()
    model = model_list[0]
    model.eval()
    class_pred = model(image_classify)
    class_nums = torch.argmax(class_pred, 1).cpu().numpy()
    # class_nums = [0]
    label_list[class_nums[0]] = label_list[class_nums[0]] + 1
    model = model_list[class_nums[0] + 1]
    model.eval()
    predict_1 = model(image)
    predict_2 = model(torch.flip(image, [-1]))
    predict_2 = torch.flip(predict_2, [-1])
    predict_3 = model(torch.flip(image, [-2]))
    predict_3 = torch.flip(predict_3, [-2])
    predict_4 = model(torch.flip(image, [-1, -2]))
    predict_4 = torch.flip(predict_4, [-1, -2])
    predict_list += (predict_1 + predict_2 + predict_3 + predict_4)
    predict_list = torch.argmax(predict_list.cpu(), 1).byte().numpy()[0]  # n x h x w


    if not os.path.exists(save_supervised_path[dataloader[4]]):
        os.makedirs(save_supervised_path[dataloader[4]])
    overlap = colorize_mask(predict_list)
    label_save_path = save_supervised_path[dataloader[4]] + '\\' + image_path.split('\\')[-1][:-4] + '_predict_' + str(
        class_nums[0]) + '.tif'
    overlap.save(label_save_path)

    segmentation_list = []
    mask = np.array(show).flatten()
    label = np.array(predict_list).flatten()
    keys = np.unique(mask)
    list_count = [0, 0, 0, 0, 0, 0]
    for indx in seg_lab:
        u_labels, hist = np.unique(label[indx], return_counts=True)
        max_number = u_labels[np.argmax(hist)]
        segmentation_list.append([indx[0], max_number, indx])

    for key in keys:
        # index_list = []
        #     print(np.where(mask==key)[0])
        for segmentation in segmentation_list:
            if segmentation[0] in np.where(mask == key)[0]:
                list_count[segmentation[1]] = list_count[segmentation[1]] + 1
        label_sata = Counter(label[np.where(mask == key)])
        if sum(list_count) == 0 or sum(label_sata.values()) == 0:
            break
        water = (list_count[5] / sum(list_count)) >= 0.6 and label_sata[5] / sum(label_sata.values()) >= 0.6
        farm = (list_count[2] / sum(list_count)) >= 0.6 and label_sata[2] / sum(label_sata.values()) >= 0.6  #
        forest = (list_count[3] / sum(list_count)) >= 0.6 and label_sata[3] / sum(label_sata.values()) >= 0.6
        meadow = (list_count[4] / sum(list_count)) >= 0.6 and label_sata[4] / sum(label_sata.values()) >= 0.6
        if water or farm or forest or meadow:
            max_number = label_sata.most_common(1)[0][0]
            label[mask == key] = max_number

        else:
            for segmentation in segmentation_list:
                label[segmentation[2]] = segmentation[1]

    label = label.reshape(512, 512)
    pos = pos_list
    [topleft_x, topleft_y, buttomright_x, buttomright_y] = pos

    if (buttomright_x - topleft_x) == 512 and (buttomright_y - topleft_y) == 512:
        # zeros[topleft_y + 128:buttomright_y - 128, topleft_x + 128:buttomright_x - 128] = label[128:384, 128:384]
        if not os.path.exists(save_segementation_path[dataloader[4]]):
            os.makedirs(save_segementation_path[dataloader[4]])
        overlap = colorize_mask(label)
        label_save_path = save_segementation_path[dataloader[4]] + '\\' + image_path.split('\\')[-1][
                                                                          :-4] + '_segmentation.tif'
        overlap.save(label_save_path)
    else:
        raise ValueError(
            "target_size!=512， Got {},{}".format(buttomright_x - topleft_x, buttomright_y - topleft_y))

    print(image_path.split('\\')[-1][:-4] + '_segmentation.tif')


class Create_Dataset(object):
    def __init__(self, root_dir, csv_file, transforms, number, unsupervised_path, csv_path):
        self.root_dir = root_dir
        self.csv_file = pd.read_csv(csv_file, header=None)
        self.transforms = transforms
        self.number = number
        self.image_lists = []
        self.unsupervised_path = unsupervised_path
        self.csv_path = csv_path

    def create_list(self):
        file_list_unsupervised = []
        seg_list = []
        for file in os.listdir(self.unsupervised_path[self.number]):
            file_list_unsupervised.append(file)
        file_list_unsupervised.sort(key=lambda x: int(re.match('\S+\_(\d+)\_cnn', x).group(1)))

        for file in os.listdir(self.csv_path[self.number]):
            seg_list.append(file)
        seg_list.sort(key=lambda x: int(re.match('\S+\_(\d+)\_seg', x).group(1)))

        for idx in range(self.csv_file.shape[0]):
            image_list = []
            filename = self.csv_file.iloc[idx, 0]
            # print(filename)
            image_path = os.path.join(self.root_dir, filename)

            image = Image.open(image_path).convert('RGB')
            image = self.transforms(image)
            img = cv2.imread(image_path)
            img = Image.fromarray(img)
            image_classify = self.transforms(img)
            pos_list = self.csv_file.iloc[idx, 1:].values.astype(
                "int")  # ---> (topleft_x,topleft_y,buttomright_x,buttomright_y)
            image_list.append(image)
            image_list.append(pos_list)
            image_list.append(image_path)
            image_list.append(image_classify)
            image_list.append(self.number)
            image_list.append(os.path.join(self.csv_path[self.number], seg_list[idx]))
            image_list.append(os.path.join(self.unsupervised_path[self.number], file_list_unsupervised[idx]))

            self.image_lists.append(image_list)
        return self.image_lists


class DenseModel(nn.Module):
    def __init__(self, pretrained_model):
        super(DenseModel, self).__init__()
        self.classifier = nn.Linear(pretrained_model.classifier.in_features, 6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.features = pretrained_model.features
        self.layer1 = pretrained_model.features._modules['denseblock1']
        self.layer2 = pretrained_model.features._modules['denseblock2']
        self.layer3 = pretrained_model.features._modules['denseblock3']
        self.layer4 = pretrained_model.features._modules['denseblock4']

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=8).view(features.size(0), -1)
        out = F.sigmoid(self.classifier(out))
        return out


def parse_args():
    parser = argparse.ArgumentParser(description="膨胀预测")
    # parser.add_argument('--classify', type=str, default= classify)
    parser.add_argument('--test-data-root', type=str,
                        default='C:\\Users\\dell\\code\\dxj\\High-Resolution-Remote-Sensing-Semantic-Segmentation-PyTorch-master\\tools\\data_512\\' + 'classify' + '_image')

    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='batch size for testing (default:16)')
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument("--model-path", type=str,
                        # default="D:\\dataset\\Graduation_project\\model\\512\\deeplabv3_version_1_final\\resnet50"
                        # default="D:\\dataset\\Graduation_project\\model\\512\\U_net_final\\resnet50"
                        # default="D:\\dataset\\Graduation_project\\model\\512_final\\FPN\\resnet50"
                        # default="D:\\dataset\\Graduation_project\\model\\512\\PSPNet\\resnet50"
                        default="D:\\dataset\\Graduation_project\\model\\512\\GCN\\resnet50"

                        )
    parser.add_argument("--model-path-scene", type=str,
                        default="D:\\dataset\\Graduation_project\\model\\resnet152_model_512_0.75\\65.638_39.pkl")
    # parser.add_argument("--model-path-scene", type=str,
    #                     default="D:\\dataset\\Graduation_project\\model\\dense_model_512\\79.374_4.pkl")
    parser.add_argument("--pred-path", type=str, default="")

    parser.add_argument("--model-name", type=list,
                        default=[])
    parser.add_argument("--save-path", type=str,
                        default="D:\\dataset\\Graduation_project\\Graduation_result\\512\\" + 'classify' + "_result_final\\resnet152_final_pre_16")
    parser.add_argument("--csv-file", type=str,
                        default=
                        [
                            r'C:\code\dxj\High-Resolution-Remote-Sensing-Semantic-Segmentation-PyTorch-master\tools\data_512\GF2_PMS1__L1A0001670888-MSS1.csv']
                        )
    args = parser.parse_args()

    return args


zeros = (6800, 7200)
h, w = zeros[0], zeros[1]
new_h, new_w = (h // 512 + 1) * 512, (w // 512 + 1) * 512  # 填充下边界和右边界得到滑窗的整数倍
zeros = (new_h + 128, new_w + 128)  # 填充空白边界，考虑到边缘数据
zeros = np.zeros(zeros, np.uint8)
model_list = []

args = parse_args()

# sc_model = DenseModel(torchvision.models.densenet169(pretrained=True))
sc_model = models.resnet152(pretrained=False)
sc_model.avgpool = nn.AdaptiveAvgPool2d(1)
sc_model.fc = nn.Linear(2048, 6)
sc_model.load_state_dict(torch.load(args.model_path_scene))
# sc_model = nn.DataParallel(sc_model, device_ids=[0])
sc_model = sc_model.cuda()

model_list.append(sc_model)
for i in range(6):
    # model = DeepLabV3(num_classes=6)
    # model = UNet(n_channels=3, n_classes=6)
    # n_blocks = '3, 4, 23, 3'
    # n_blocks = [int(b) for b in n_blocks.split(',')]
    # model = FPN(num_classes=6,num_blocks=n_blocks)
    # model = PSPNet(6, 8)
    model = FCN_GCN(6)
    state_dict = torch.load(args.model_path + args.model_name[i])
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    # model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    # model = nn.DataParallel(model, device_ids= list[1])
    model_list.append(model)
save_path_number = 0
save_unsupervised_path = []
save_segementation_path = []
save_supervised_path = []
save_csv_path = []
for k in range(15):
    save_unsupervised = r"D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_" + str(
        k + 1) + "\image_unsupervised_16\image_unsupervised"
    save_unsupervised_path.append(save_unsupervised)
    save_segementation = r"D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_" + str(
        k + 1) + "\image_unsupervised_16\image_0.6_GCN"
    save_segementation_path.append(save_segementation)
    save_supervised = r"D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_" + str(
        k + 1) + "\image_supervised_GCN"
    save_supervised_path.append(save_supervised)
    save_csv = r"D:\dataset\Graduation_project\Graduation_result\512_final_GID\image_val_" + str(
        k + 1) + "\image_felzenszwalb"
    save_csv_path.append(save_csv)


def main():
    classify_list = [ ]

    csv_file_list = [ ]
    for i in range(len(classify_list)):
        args.test_data_root = "C:\\Users\\dell\\code\\dxj\\High-Resolution-Remote-Sensing-Semantic-Segmentation-PyTorch-master\\tools\\data_512\\" + \
                              classify_list[i]
        args.csv_file = csv_file_list[i]
        label_save_path = save_segementation_path[i] + ".png"
        dataset = Create_Dataset(root_dir=args.test_data_root,
                                 csv_file=args.csv_file,
                                 transforms=img_transform, number=i,
                                 unsupervised_path=save_unsupervised_path,
                                 csv_path=save_csv_path
                                 ).create_list()
        pool = Pool(processes=2)
        pool.map(snapshot_forward, dataset)
        csv_file = pd.read_csv(args.csv_file, header=None)
        for idx in tqdm(range(csv_file.shape[0])):

            filename = csv_file.iloc[idx, 0]
            # print(filename)
            image_path = os.path.join(save_segementation_path[i], filename[:-4] + '_segmentation.tif')

            dataset = gdal.Open(image_path)
            label = dataset.GetRasterBand(1).ReadAsArray()

            pos_list = csv_file.iloc[idx, 1:].values.astype("int")
            [topleft_x, topleft_y, buttomright_x, buttomright_y] = pos_list
            if (buttomright_x - topleft_x) == 512 and (buttomright_y - topleft_y) == 512:
                zeros[topleft_y + 128:buttomright_y - 128, topleft_x + 128:buttomright_x - 128] = label[128:384,
                                                                                                  128:384]

        h, w = zeros.shape
        png = zeros[128:h - 128, 128:w - 128]
        zero = (6800, 7200)  # 去除补全512整数倍时的右下边界
        png = png[:zero[0], :zero[1]]

        overlap = colorize_mask(png)
        overlap.save(label_save_path)


if __name__ == '__main__':
    main()
