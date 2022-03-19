if __name__ == '__main__':
    import utils.utils as ut

    ut.startFlatten(root_path='../sources/dataset/', dir_list=['12'], log_path='log.txt',
                 save_path='../sources/dataset/12/done/', deg=1, mbKSize=9)

    # # img = cv2.imread('../sources/dataset/12/14-181.jpg', flags=0)
    # img = cv2.imread('14_122.jpg', flags=0)
    # ret = ut.standardization(img)
    # region = ut.find_max_region(ret, 9, denoiseScheme='default')
    #
    # y, _ = ut.surfaceFitting(ret, deg=2, mbSize=9)
    # fitted_location_img = np.copy(img)
    # original_location_img = np.copy(img)
    # for i in range(fitted_location_img.shape[1]):
    #     fitted_location_img[y[i]][i] = 255
    # for i in range(len(_)):
    #     k = _[i]
    #     original_location_img[k[0]][k[1] - 1] = 255
    # flattened_img = ut.flatten(img, [512 - i for i in y])
    #
    # merge = np.hstack((img, ret, region, original_location_img, fitted_location_img, flattened_img))
    # # cv2.imwrite('14_181.jpg', merge)
    # # cv2.imwrite('14_122_single.jpg', flattened_img)
    # cv2.imwrite('14_122_1.jpg', flattened_img)
