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


if __name__ == '__main__':
    import utils.utils as ut
    import os
    import cv2

    root_path = '../sources/dataset/'
    dir_list = os.listdir(root_path)
    log_path = 'log.txt'

    with open(log_path, 'w+') as f:
        for dir in dir_list:
            img_root_path = root_path + dir + '/'
            img_list = os.listdir(img_root_path)
            print(dir + '开始：')
            f.write(dir + '开始：')
            f.write('\n')
            for idx, img_name in enumerate(img_list):
                try:
                    if img_name == 'save':
                        continue
                    img_path = img_root_path + img_name
                    save_path = img_root_path + 'save/'
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    img = cv2.imread(img_path, flags=0)
                    ret = ut.standardization(img)
                    y = ut.surfaceFitting(ret)
                    img = ut.flatten(img, [512 - i for i in y])
                    img = ut.cropImg(img, 50, 0, 10, 40)
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
