import os

import cv2
import torch
from torch.utils.data import DataLoader


def surfaceFitting(img, k, deg=3, plot=False):
    """
    表面拟合
    Parameters
    ----------
        img：数据类型为numpy.narray
            图片数据

        k：数据类型为int
            提取拟合数据点用到的阈值k
            将图片中低于k的灰度值置为0。

        deg:数据类型为int
            拟合使用的多项式的阶数
            默认为3

        plot：数据类型为boolean
            是否展示绘制拟合的结果。
            默认为False

    Returns
    -------
        surfacePosition:数据类型为numpy.array
            表面位置数组
            其中的每一个数 value,有这样一个关系：col_index = img.shape[1] - value

        ret:数据类型为numpy.narray
            拟合点的图片

    Examples
    -------
        y, _ = surfaceFitting(img, k, deg)
        surfacePosition = np.array([512 - i for i in y])
        flattenImg = flatten(img, surfacePosition)

    """
    ret = np.copy(img)
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            if ret[i][j] <= (k):
                ret[i][j] = 0
    temp = []
    for i in range(ret.shape[1]):
        fd = np.flipud(ret[:, i])
        j = ret.shape[1]
        while j > 0:
            if j != 0 and j < 500 and fd[j] != 0:
                temp.append((512 - j, i + 1))
                break
            j -= 1
    temp = np.array(temp)
    X = temp[:, 1]
    y = temp[:, 0]
    z1 = np.polyfit(X, y, deg)
    p1 = np.poly1d(z1)
    X_test = [i + 1 for i in range(ret.shape[1])]
    surfacePosition = np.array(p1(X_test), dtype=np.int32)
    if plot:
        pts1 = np.concatenate((X_test, surfacePosition)).reshape((2, 512)).T
        test = cv2.polylines(np.copy(img), [pts1], False, color=(255, 0, 0))
        cv2.imshow('press to continue', test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return surfacePosition, ret


def flatten(img, surfacePosition):
    """
    展平图片
    Parameters
    ----------
        img:数据类型为numpy.narray
            图片数据

        surfacePosition：数据类型为numpy.array
            表面位置索引数组
            由列向量索引组成，长度等于图片的列向量数量
            例如np.array([224, 212, 264, 203 ....，123])

    Returns
    -------
        ret：数据类型为numpy.narray
            展平后的图片
    """
    ret = np.zeros(shape=img.shape, dtype=np.uint8)
    imax = np.max(surfacePosition)
    imin = np.min(surfacePosition)
    mid = int((imax + imin) / 2)
    for idx, value in enumerate(surfacePosition):
        diff = np.abs(value - mid)
        temp = img[:, idx]
        if value > mid:
            k = diff
            for j in range(temp.size - diff):
                ret[k][idx] = temp[j]
                k += 1
        else:
            k = 0
            for j in range(diff, temp.size):
                ret[k][idx] = temp[j]
                k += 1
    return ret


def denoise(img, n, kernel_size=3):
    """
    图片去噪
    Parameters
    ----------
        img:数据类型为numpy.narray
            图片数据

        n:数据类型为int
            使用前n行的均值和标准差进行阈值去噪

        kernel_size：数据类型为int
            中值滤波核的大小

    Returns
    -------
        ret:数据类型为numpy.narray
            返回图片

    """
    ret = cv2.medianBlur(img, kernel_size)
    k = []
    num = 0
    sum = 0
    for i in range(n):
        for j in range(ret.shape[1]):
            if ret[i][j] <= (5):
                continue
            sum += ret[i][j]
            k.append(ret[i][j])
            num += 1
    ave = num / sum
    sum = 0
    for value in k:
        sum += np.power(value - ave, 2)
    sd = np.sqrt(sum / num)
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            if ret[i][j] <= sd + ave:
                ret[i][j] = 0
    return ret


def cropImg(img, top=0, button=0, left=0, right=0):
    """
    裁剪图片
    Parameters
    ----------
        img:数据类型为numpy.narray
            图片数据

        top:数据类型为int
            裁剪掉距离顶部的k行
            默认值为0

        button:数据类型为int
            裁剪掉距离底部的k行
            默认值为0

        left:数据类型为int
            裁剪掉距离左边的k列
            默认值为0

        right:数据类型为int
            裁剪掉距离右边的k列
            默认值为0

    Returns
    -------
        ret:数据类型为numpy.narray
            裁剪完的图片

    Examples
    --------
        img = cv2.imread('7-462.jpg', flags=0)
        ret = cropImg(img, 60)
    """
    ret = img[top:img.shape[0] - button, left:img.shape[1] - right]
    return ret


def copyD2D(srcDir: str, dstDir: str):
    """
    將一個文件夾中的所有圖片copy到制定目錄
    Parameters
    ----------
        srcDir: String
            源目錄

        dstDir: String
            目標目錄

    """
    from shutil import copy
    srcNames = os.listdir(srcDir)
    for item in srcNames:
        copy(srcDir + item, dstDir)
        print(dstDir + item + ' done!')


class TestModel:
    """
    模型測試的類
    """
    def __init__(self, model, modelLocation, strict=True):
        model.load_state_dict(torch.load(modelLocation), strict=strict)
        self.model = model

    def testSingleImg(self, img):
        """
        以單張圖片的格式來測試網絡
        Parameters
        ----------
        img:數據類型爲numpy.narray
            圖片數據

        Returns
        -------
            預測類別

        Examples
        -------
            img = cv2.imread('../sources/dataset/apis-mellifera/apis-mellifera_2_a0020469.jpg')
            transform = torchvision.transforms.ToTensor()
            TestModel(ResNet50(ResidualBlock, 3, 151), 'model/net_107.pth').testSingleImg(transform(img).view(1, 3, 224, 224))
        """
        out = self.model(img)
        _, prediction = torch.max(out, 1)
        return prediction.numpy()[0]

    def testDataLoader(self, dataLoader: DataLoader):
        """
        以DataLoader的形式來測試網絡預測的準確度，命令行會打印出準確率
        Parameters
        ----------
        dataLoader: DataLoader
            DataLoader類型的數據集

        Examples
        -------
        TestModel(ResNet50(ResidualBlock, 3, 151), 'model/net_107.pth').testDataLoader(valloader)

        """
        correct = 0
        total = 0
        print('正在计算准确率...')
        for i, data in enumerate(dataLoader):
            inputs, labels = data
            outputs = self.model(inputs)
            _, prediction = torch.max(outputs, 1)
            total += labels.size(0)
            correct += prediction.eq(labels.data).cpu().sum()
        print('Total Sample:{}, True Number:{}, Acc:{:.3f}%'.format(total, correct, 100. * correct / total))

    def getFeatureVector(self, img):
        """
        将单张图片转换成特征向量
        Parameters
        ----------
            img: 数据类型为numpy.narray
                圖片數據，數組格式爲(1, channel_num, H, W)

        Returns
        -------
            返回圖片的特徵向量
        """
        return self.model(img).detach().numpy().flatten()


def getAllFeatureVector(rootPath: str,
                        model,
                        modelLocation: str,
                        transform,
                        txtRootPath: str = ''):
    """
    將制定目錄下的各類圖片轉換成向量的形式，以便操作。
    特徵提取是由CNN完成
    Parameters
    ----------
        rootPath: Sting
            图片根目录。
            如果你的其中一张图片路径是../sources/dataset/apple/apple1.jpg，那么rootPath = '../sources/dataset/'
            
        model:
            网络模型。
            需要继承nn.Moudel

        modelLocation: String
            预训练后的模型参数文件地址。

        transform:
            需要对输入的图片做的预处理
            比如torchvision.transforms.ToTensor()是将图片转换成tensor格式

        txtRootPath:String
            图片向量保存的根目录。

    Examples
    -------
        model = ResNet50Regression(1)
        modelLocation = 'model/net_22.pth'
        rootPath = '../sources/dataset/'
        transform = torchvision.transforms.ToTensor()
        getAllFeatureVector(rootPath=rootPath, model=model, modelLocation=modelLocation, transform=transform)

    """
    import numpy as np
    temp = []
    names = os.listdir(rootPath)
    print('wait a minute...')
    try:
        for name in names:
            imgNames = os.listdir(rootPath + name + '/')
            for imgName in imgNames:
                img = cv2.imread(rootPath + name + '/' + imgName, flags=0)
                k = TestModel(model=model, modelLocation=modelLocation, strict=False) \
                    .getFeatureVector(transform(img).view(1, 1, 224, 224))
                temp.append(k)
            temp = np.array(temp)
            np.savetxt(txtRootPath + name + '.txt', temp, '%f', delimiter=',')
            print(name + '.txt ' + 'saved successfully! The number of records is {}'.format(temp.shape[0]))
            temp = []
    except:
        print('Save failed')
