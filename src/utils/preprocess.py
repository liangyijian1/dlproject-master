import os

import cv2
import torch
from torch.utils.data import DataLoader


def surfaceFitting(img, deg=3, plot=False):
    import numpy as np
    region = find_max_region(img, 13)
    col = region.shape[1]
    row = region.shape[0]
    location = []
    for i in range(col):
        temp = region[:, i]
        j = row - 1
        while j > 0:
            if temp[j] > 0:
                location.append(j)
                break
            j -= 1
    temp = []
    for i in range(len(location)):
        temp.append((location[i], i + 1))
    temp = np.array(temp)
    X = temp[:, 1]
    y = temp[:, 0]
    z1 = np.polyfit(X, y, deg)
    p1 = np.poly1d(z1)
    X_test = [i + 1 for i in range(img.shape[1])]
    surfacePosition = np.array(p1(X_test), dtype=np.int32)
    if plot:
        pts1 = np.concatenate((X_test, surfacePosition)).reshape((2, 512)).T
        test = cv2.polylines(np.copy(img), [pts1], False, color=(255, 0, 0))
        cv2.imshow('press to continue', test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return surfacePosition


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
    import numpy as np
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
    import numpy as np
    ret = np.copy(img)
    k = []
    num = 0
    sum = 0
    for i in range(n):
        for j in range(ret.shape[1]):
            if ret[i][j] <= 4:
                continue
            sum += ret[i][j]
            k.append(ret[i][j])
            num += 1
    ave = sum / num
    sum = 0
    for value in k:
        sum += np.power(value - ave, 2)
    sd = np.sqrt(sum / num)
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            if ret[i][j] <= sd + ave:
                ret[i][j] = 0
    for i in range(30):
        for j in range(ret.shape[1]):
            ret[i, j] = 0
            ret[511 - i, j] = 0
    ret = cv2.medianBlur(ret, ksize=kernel_size)
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


def getFeatureVectorPlus(img,
                         img_save_path: str,
                         res_save_path: str,
                         k: int,
                         deg: int,
                         transform,
                         model,
                         model_dict_path: str,
                         crop: list = [0, 0, 0, 0],
                         strict: bool = False,
                         n: int = 5,
                         kernel_size: int = 3,
                         progress: bool = False):
    """
    提取一张不经过任何处理图片的特征向量
    Parameters
    ----------
    img
    img_save_path
    res_save_path
    k
    deg
    transform
    model
    model_dict_path
    crop
    strict
    n
    kernel_size
    progress

    Returns
    -------
    Examples
    --------
       getFeatureVectorPlus(img=img,
                            img_save_path='test/img/te-12-262.jpg',
                            res_save_path='test/res/te-12-262.txt',
                            k=190,
                            deg=3,
                            transform=torchvision.transforms.ToTensor(),
                            model=ResNet50Regression(1),
                            model_dict_path='model/net_22.pth',
                            crop=[60, 0, 30, 30]
                            )
    """
    import cv2
    import numpy as np
    surface, _ = surfaceFitting(img, k=k, deg=deg, plot=progress)
    if progress:
        cv2.imshow('PRESS TO CONTINUE', _)
        cv2.waitKey(0)
    surfacePosition = np.array([img.shape[1] - i for i in surface])
    ret = flatten(img, surfacePosition)
    ret = denoise(ret, n=n, kernel_size=kernel_size)
    ret = cropImg(ret, crop[0], crop[1], crop[2], crop[3])
    cv2.imwrite(img_save_path, ret)
    feature = TestModel(model, model_dict_path, strict=strict) \
        .getFeatureVector(transform(np.resize(ret, (224, 224))).view(1, 1, 224, 224))
    np.savetxt(res_save_path, feature.reshape(1, -1), fmt='%f', delimiter=',')


def make_labels(root_path: str, save_path: str, label_location: str):
    """
    为图片特征向量最后一列中加上对应标签
    Parameters
    ----------
    root_path: String
        图片向量文件存放的根路径
        root_path/feature.txt

    save_path: String
        加上标签后文件存放的位置

    label_location: String
        label文件的路径

    Notes
    -----
    label是以json格式保存在可读文件中, 键名是图片向量的文件名， 值为标签
    比如apple这个类的图片向量被保存到apple.txt中，并且它对应的标签为1，
    label中就是 {"apple": 1}

    """
    import os
    import json
    import numpy as np
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(label_location, 'r') as f:
        s1 = json.load(f)
        for key, label in s1.items():
            path_key = root_path + key + '.txt'
            if os.path.exists(path_key):
                with open(path_key, 'r') as f1:
                    content = f1.readlines()
                    for i in range(len(content)):
                        content[i] = content[i].strip().split(',')
                    content = np.array(content).astype(np.float64)
                    content = np.insert(content, content.shape[1], s1[key], axis=1)
                    np.savetxt(save_path + key + '-done.txt', content, '%f', delimiter=',')


class MarkLabel:
    """
    OCT图像分层标记
    """

    def __init__(self, win_name: str):
        cv2.namedWindow(win_name, 0)
        self.win_name = win_name
        print('标记完后按下Enter保存，退出按ESC键\n')

    def on_mouse_pick_points(self, event, x, y, flags, param):
        """
        鼠标事件回调函数
        """
        if flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.drawMarker(param, (x, y), 255, markerSize=1)

    def markLabel(self, img, save_path: str = None):
        """
        分层标记点
        Parameters
        Examples
        --------
        MarkLabel('show').markLabel(image, 'test/img/262label.jpg')
        """
        cv2.setMouseCallback(self.win_name, self.on_mouse_pick_points, img)
        while True:
            cv2.imshow(self.win_name, img)
            key = cv2.waitKey(30)
            if key == 27:  # ESC
                break
            elif key == 13:  # enter
                if save_path is None:
                    print('没有给定文件路径!\n')
                    continue
                cv2.imwrite(save_path, img)
                print('保存成功\n')
        cv2.destroyAllWindows()
