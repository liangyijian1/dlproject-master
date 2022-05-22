import os

import cv2
import torch.utils.data
from torch.utils.data.dataset import T_co


def get_txt(imgPath, label, txtPath):
    img_list = os.listdir(imgPath)
    with open(txtPath, 'w') as f:
        for item in img_list:
            f.write(item + ' ' + str(label))
            f.write('\n')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, txtPath, imgPath, label=None, transform=None):
        super(MyDataset, self).__init__()
        if not os.path.exists(txtPath):
            get_txt(imgPath=imgPath, label=label, txtPath=txtPath)
        imgs = []
        with open(txtPath, 'r') as f:
            for item in f.readlines():
                if item == '\n':
                    continue
                line = item.strip("\n").split(' ')
                imgs.append((line[0], torch.tensor(int(float(line[1])), dtype=torch.int64)))
            self.imgs = imgs
            self.imgPath = imgPath
            self.transform = transform

    def __getitem__(self, index) -> T_co:
        pic, label = self.imgs[index]
        img = cv2.imread(self.imgPath + pic, 0)
        rows = img.shape[0]
        cols = img.shape[1]
        if self.transform is not None:
            img = self.transform(img).view((1, 1, rows, cols))
        return img, label

    def __len__(self):
        return len(self.imgs)


def img_preprocess_test(basePath):
    pathsNames = os.listdir(basePath)
    for idx, pathName in enumerate(pathsNames):
        imgNames = os.listdir(basePath + pathName)
        for k, item in enumerate(imgNames):
            os.rename(basePath + pathName + '/' + item,
                      basePath + pathName + '/' + pathName + '-' + k.__str__() + '-' + idx.__str__() + '.jpg')


def load_local_dataset(all_dataset, batch_size):
    pre_size = int(0.95 * len(all_dataset))
    val_size = len(all_dataset) - pre_size
    train_size = int(0.8 * pre_size)
    test_size = pre_size - train_size
    pre_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [pre_size, val_size],
                                                             torch.Generator().manual_seed(42))
    train_dataset, test_dataset = torch.utils.data.random_split(pre_dataset, [train_size, test_size],
                                                                torch.Generator().manual_seed(42))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, valloader


# if __name__ == '__main__':
# img_preprocess_test('../sources/dataset/test/')
# img_preprocess('../sources/6/', '6', 185, 3.5)
# img_preprocess('../sources/9/', '9', 180, 2.5)

# imgPaths = ['../sources/dataset/test/0/', '../sources/dataset/test/4/', '../sources/dataset/test/8/',
#             '../sources/dataset/test/12/', '../sources/dataset/test/16/']
# txtPaths = ['../src/0.txt', '../src/4.txt', '../src/8.txt', '../src/12.txt', '../src/16.txt']
# k = 0
# for i in range(len(imgPaths)):
#     dataset = MyDataset(imgPath=imgPaths[i], txtPath=txtPaths[i], label=k)
#     k += 4

# pathsNames = os.listdir(imgPaths[0])
# for idx, pathName in enumerate(pathsNames):
#     imgNames = os.listdir(imgPaths[0] + pathName)
#     label = imgNames[0].split('-')[-1].split('.')[0]
#     dataset = MyDataset(imgPath=imgPaths[0] + pathName + '/', txtPath=txtPaths[0] + pathName + '.txt', label=label)

# txtNames = os.listdir(txtPaths[0])
# for txtName in txtNames:
#     appendFileName(txtPaths[0] + txtName, txtPaths[0] + 'img.txt')

# pathsNames = os.listdir(imgPaths[0])
# for item in pathsNames:
#     copyD2D(imgPaths[0] + item + '/', '../sources/dataset/img/')
