import os

import numpy as np
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
    '''
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
    '''
    ret = img[top:img.shape[0] - button, left:img.shape[1] - right]
    return ret


def preprocess(img, k, deg):
    y, _ = surfaceFitting(img, k, deg)
    surfacePosition = np.array([512 - i for i in y])
    flattenImg = flatten(img, surfacePosition)
    flattenImg = denoise(flattenImg, 3)
    flattenImg = cropImg(flattenImg, 60)
    return flattenImg


def appendFileName(src, dst):
    with open(src, 'r') as f:
        content = f.readlines()
        with open(dst, 'a+') as c:
            c.write('\n')
            for item in content:
                c.write(item)
            c.close()
        f.close()


def copyD2D(srcDir, dstDir):
    from shutil import copy
    srcNames = os.listdir(srcDir)
    for item in srcNames:
        copy(srcDir + item, dstDir)
        print(dstDir + item + ' done!')


class TestModel:

    def __init__(self, model, modelLocation, strict=True):
        model.load_state_dict(torch.load(modelLocation), strict=strict)
        self.model = model

    def testSingleImg(self, img):
        """

        Parameters
        ----------
        img

        Returns
        -------
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
        '''

        Parameters
        ----------
        dataLoader

        Returns
        -------
        Examples
        -------
        TestModel(ResNet50(ResidualBlock, 3, 151), 'model/net_107.pth').testDataLoader(valloader)

        '''
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

        Parameters
        ----------
            img:
                (1, channel_num, H, W)

        Returns
        -------

        """
        return self.model(img)

