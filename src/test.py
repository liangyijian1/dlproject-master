import cv2
import numpy as np
import numpy.typing

import utils.preprocess as pr


# noinspection PyShadowingNames,SpellCheckingInspection,PyShadowingBuiltins
def standardization(img, ksize=15):
    ret = np.copy(img)
    temp = cv2.medianBlur(img, ksize=ksize)
    row = ret.shape[0]
    col = ret.shape[1]
    max_temp = np.max(temp)
    max = np.max(ret)
    min = np.min(ret)
    Im = 1.05 * max_temp
    for i in range(row):
        for j in range(col):
            if ret[i][j] <= Im:
                ret[i][j] = Im * (ret[i][j] - min) / (max - min)
            else:
                ret[i][j] = Im
    return ret


# noinspection PyShadowingNames,PyShadowingBuiltins
def confirm(pre_estimate_location: numpy.typing.NDArray,
            threshold1: int,
            threshold2: int):
    """

    Parameters
    ----------
    pre_estimate_location: narray
        预估计位置数组
    threshold1
    threshold2

    Returns
    -------

    """
    import operator
    from functools import reduce
    length = len(pre_estimate_location)
    i, j = 0, 0
    fragments: list = []
    while j != length:
        while np.abs(pre_estimate_location[j + 1] - pre_estimate_location[j]) < threshold1:
            j += 1
            if j == length - 1:
                break
        clips = pre_estimate_location[i:j+1].tolist()
        fragments.append(clips)
        i = j = j + 1
    i = 0
    while i < len(fragments):
        if i == len(fragments) - 1:
            break
        current = fragments[i]
        next = fragments[i + 1]
        current_len = len(current)
        if next[0] - current[current_len - 1] > threshold2:
            fragments[i + 1] = [int(-(j / j)) for j in fragments[i + 1]]
            i += 1
        i += 1
    fragments = reduce(operator.add, fragments)
    return fragments


if __name__ == '__main__':
    # reg = Regression()
    # X, y = ut.get_data('res/features/done/img.txt')
    # # dt, dt_parm, dt_score = reg.dtRegression(X, y, 1, 50, cv_num=5, n_iter=50)
    # # print(dt_score)
    # a, b, model = reg.mlRegression(X, y)
    # # y_pre = a.predict(content.reshape(1, -1))
    # # print(y_pre)
    #
    # # ut.saveModel('dt_model.pkl', dt)
    # # y_pre = dt.predict(content.reshape(1, -1))
    # # print(y_pre)
    # img = cv2.imread('../sources/dataset/12/13-205.jpg', flags=0)
    # pr.surfaceFitting(img, 185, 3, True)
    # # pr.getFeatureVectorPlus(img=img,
    # #                         img_save_path='test/img/te-16-225.jpg',
    # #                         res_save_path='test/res/te-16-225.txt',
    # #                         k=200,
    # #                         deg=3,
    # #                         transform=torchvision.transforms.ToTensor(),
    # #                         model=ResNet50Regression(1),
    # #                         model_dict_path='model/net_22.pth',
    # #                         crop=[60, 0, 30, 30],
    # #                         progress=True
    # #                         )
    # with open('test/res/te-16-225.txt', 'r') as f:
    #     feature = f.readline().strip().split(',')
    #     feature = np.array(feature).astype(np.float64)
    # y_pre = a.predict(feature.reshape(1, -1))
    # print(y_pre)
    img = cv2.imread('../sources/dataset/9/4-025.jpg', flags=0)
    ret = standardization(img)
    de = pr.denoise(ret, 10, kernel_size=11)
    sobel = cv2.Sobel(de, cv2.CV_64F, 0, 1)
    row = img.shape[0]
    col = img.shape[1]
    k = []
    for i in range(col):
        temp = sobel[:, i]
        idx = np.argmin(temp)
        k.append(idx)
    k = np.array(k)
    filter = confirm(k, 5, 10)
    for i in range(col):
        if k[i] == -1:
            continue
        de[k[i], i] = 255
    for i in range(col):
        if filter[i] == -1:
            continue
        ret[filter[i], i] = 255
    merge = np.hstack((img, ret, de))
    cv2.imshow('1', merge)
    cv2.waitKey(0)
