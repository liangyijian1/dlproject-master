# # noinspection PyShadowingNames,PyShadowingBuiltins
# def confirm(pre_estimate_location: numpy.typing.NDArray,
#             threshold1: int,
#             threshold2: int):
#     """
#
#     Parameters
#     ----------
#     pre_estimate_location: narray
#         预估计位置数组
#     threshold1
#     threshold2
#
#     Returns
#     -------
#
#     """
#     import operator
#     from functools import reduce
#     length = len(pre_estimate_location)
#     i, j = 0, 0
#     fragments: list = []
#     while j != length:
#         while np.abs(pre_estimate_location[j + 1] - pre_estimate_location[j]) < threshold1:
#             j += 1
#             if j == length - 1:
#                 break
#         clips = pre_estimate_location[i:j + 1].tolist()
#         fragments.append(clips)
#         i = j = j + 1
#     i = 0
#     while i < len(fragments):
#         if i == len(fragments) - 1:
#             break
#         current = fragments[i]
#         next = fragments[i + 1]
#         current_len = len(current)
#         if next[0] - current[current_len - 1] > threshold2:
#             fragments[i + 1] = [int(-(j / j)) for j in fragments[i + 1]]
#             i += 1
#         i += 1
#     fragments = reduce(operator.add, fragments)
#     return fragments
import numpy as np


def startFlatten(root_path: str, dir_list: list, log_path: str, save_path: str, deg: int = 3, mbKSize: int = 11,
                 denoiseScheme: str = 'default'):
    with open(log_path, 'a+') as f:
        for dir in dir_list:
            img_root_path = root_path + dir + '/failed/'
            img_list = os.listdir(img_root_path)
            print(dir + '开始：')
            f.write(dir + '开始：')
            f.write('\n')
            for idx, img_name in enumerate(img_list):
                try:
                    if img_name[-3:] != 'jpg':
                        continue
                    img_path = img_root_path + img_name
                    # save_path = img_root_path + 'save/'
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    img = cv2.imread(img_path, flags=0)
                    ret = ut.standardization(img)
                    y, _ = ut.surfaceFitting(ret, deg=deg, mbSize=mbKSize, denoiseScheme=denoiseScheme)
                    img = ut.flatten(img, [512 - i for i in y])
                    img = ut.cropImg(img, 60, 0, 10, 40)
                    cv2.imwrite(save_path + img_name, img)
                    print('total: {} current: {} '.format(len(img_list) - 1, idx + 1) + img_name + ' done')
                    f.write('total: {} current: {} '.format(len(img_list) - 1, idx + 1) + img_name + ' done')
                    f.write('\n')
                except Exception as e:
                    print('total: {} current: {} '.format(len(img_list) - 1, idx + 1) + img_name + ' failed')
                    f.write('total: {} current: {} '.format(len(img_list) - 1,
                                                            idx + 1) + img_name + ' failed' + 'reason: ' + e.__str__())
                    f.write('\n')
            print(dir + '结束\n')
            f.write(dir + '结束\n')
            f.write('\n')


if __name__ == '__main__':
    import utils.utils as ut
    import os
    import cv2

    # startFlatten('../sources/dataset/', ['12'], 'log.txt', '../sources/dataset/12/done/', 1, 5)

    img = cv2.imread('../sources/dataset/12/13-330.jpg', flags=0)
    ret = ut.standardization(img)
    region = ut.find_max_region(ret, 11, denoiseScheme='default')

    y, _ = ut.surfaceFitting(img, deg=1, mbSize=11)
    fitted_location_img = np.copy(img)
    original_location_img = np.copy(img)
    for i in range(fitted_location_img.shape[1]):
        fitted_location_img[y[i]][i] = 255
        original_location_img[_[i]][i] = 255
    flattened_img = ut.flatten(img, [512 - i for i in y])

    merge = np.hstack((img, ret, region, original_location_img, fitted_location_img, flattened_img))
    cv2.imwrite('merge5.jpg', merge)
