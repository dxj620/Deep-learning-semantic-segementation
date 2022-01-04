from torch.utils.data import Dataset
import os
from PIL import Image
from class_names import five_classes
from torchvision import transforms
import numpy as np
import torch


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
mask_transform = MaskToTensor()


class RSDataset(Dataset):
    def __init__(self, root=None, mode=None, img_transform=img_transform, mask_transform=mask_transform,
                 sync_transforms=None):
        # 数据相关
        self.class_names = five_classes()
        self.mode = mode
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.sync_transform = sync_transforms
        self.sync_img_mask = []
        key_word = 'patches'
        if mode == "src":
            img_dir = os.path.join(root, 'rgb')
            mask_dir = os.path.join(root, 'label')
        else:
            for dirname in os.listdir(root):
                # 舍弃特定训练子集
                if 'ignore' in dirname:
                    continue
                # 避免读取非训练集目录
                if not key_word in dirname:
                    continue

            img_dir = os.path.join(root, 'rgb')
            mask_dir = os.path.join(root, 'label')
        for img_filename in os.listdir(img_dir):
            index = img_filename.split('_')[-1].split('.')[0]
            # print(img_filename)
            img_mask_pair = (os.path.join(img_dir, img_filename),
                             os.path.join(mask_dir, img_filename.replace("_" + str(index) + ".jpg",
                                                                         "_label_" + str(index) + ".png") \
                                          .replace("_" + str(index) + ".jpg", "_label_" + str(index) + ".png")))
            # img_mask_pair = (os.path.join(img_dir, img_filename),
            #                  os.path.join(mask_dir, img_filename.replace("RGB_" + str(index) + ".jpg",
            #                                                              "RGB_label_" + str(index) + ".png") \
            #                               .replace("RGB_" + str(index) + ".jpg", "RGB_label_" + str(index) + ".png")))
            # img_mask_pair = (os.path.join(img_dir, img_filename),
            #                  os.path.join(mask_dir, img_filename))
            # print(img_mask_pair)
            self.sync_img_mask.append(img_mask_pair)
        # print(self.sync_img_mask)
        if (len(self.sync_img_mask)) == 0:
            print("Found 0 data, please check your dataset!")

    def __getitem__(self, index):
        img_path, mask_path = self.sync_img_mask[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        mask[(mask != 76) & (mask != 150) & (mask != 179) & (mask != 226) & (mask != 29)] = 0
        mask[mask == 76] = 1
        mask[mask == 150] = 2
        mask[mask == 179] = 3
        mask[mask == 226] = 4
        mask[mask == 29] = 5

        # list = [2, 2, 2, 2, 3, 3, 4, 4, 1, 1, 1, 6, 5, 5, 5, 0]
        # for i in range(len(list)):
        #     mask[mask == i] = list[i]
        mask = Image.fromarray(mask)
        # transform
        if self.sync_transform is not None:
            img, mask = self.sync_transform(img, mask)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.sync_img_mask)

    def classes(self):
        return self.class_names


if __name__ == "__main__":
    pass
