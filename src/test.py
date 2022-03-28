if __name__ == '__main__':


    pass
    # 机器学习模型训练过程
    # regr = Regression()
    #
    # X, y = get_data('./res/vector/vector.txt')
    # model, parm, score = regr.rfRegression(X, y, 1, 200, 1, 30, 1, 20, cv_num=10, n_iter=10)
    # # 直接使用predict()函数进行预测
    # y_pre = model.predict(X)
    # # 使用utils.py中的saveModel()函数将模型保存到本地
    # saveModel('model/ml_model/rf.pkl', model)
    # print(r2_score(y, y_pre).__str__() + '   ' + score.__str__())

    # 使用机器学习模型验证
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Resize(size=(224, 224)),
    # ])
    # img_9 = cv2.imread('../sources/dataset/test/9-3.jpg', 0)
    # ret9 = standardization(img_9)
    # img_9 = denoise(ret9, 30, 3, 15)
    #
    # img_0 = cv2.imread('../sources/dataset/test/0-1.jpg', 0)
    # ret0 = standardization(img_0)
    # img_0 = denoise(ret0, 30, 3, 15)
    #
    # img_3 = cv2.imread('../sources/dataset/test/3-2.jpg', 0)
    # ret3 = standardization(img_3)
    # img_3 = denoise(ret3, 30, 3, 15)
    #
    # img_6 = cv2.imread('../sources/dataset/test/6-2.jpg', 0)
    # ret6 = standardization(img_6)
    # img_6 = denoise(ret6, 30, 3, 15)
    #
    # img_12 = cv2.imread('../sources/dataset/test/12-3.jpg', 0)
    # ret12 = standardization(img_12)
    # img_12 = denoise(ret12, 30, 3, 15)
    #
    # feature_0 = TestModel(ResNet50Regression(1), 'model/cnn_model/net_21.pth', strict=False) \
    #     .getFeatureVector(transform(img_0).view(1, 1, 224, 224))
    # feature_3 = TestModel(ResNet50Regression(1), 'model/cnn_model/net_21.pth', strict=False) \
    #     .getFeatureVector(transform(img_3).view(1, 1, 224, 224))
    # feature_6 = TestModel(ResNet50Regression(1), 'model/cnn_model/net_21.pth', strict=False) \
    #     .getFeatureVector(transform(img_6).view(1, 1, 224, 224))
    # feature_9 = TestModel(ResNet50Regression(1), 'model/cnn_model/net_21.pth', strict=False) \
    #     .getFeatureVector(transform(img_9).view(1, 1, 224, 224))
    # feature_12 = TestModel(ResNet50Regression(1), 'model/cnn_model/net_21.pth', strict=False) \
    #     .getFeatureVector(transform(img_12).view(1, 1, 224, 224))
    # dt_pre = loadModel('model/ml_model/dt.pkl').predict(feature_9.reshape(1, -1))
    # knn_pre = loadModel('model/ml_model/knn.pkl').predict(feature_9.reshape(1, -1))
    # rf_pre = loadModel('model/ml_model/rf.pkl').predict(feature_9.reshape(1, -1))
    # svm_pre = loadModel('model/ml_model/svm.pkl').predict(feature_9.reshape(1, -1))
    # print('Decison Tree回归预测的时间是 {}\nKnn回归预测的时间是 {}\nRandom Forest回归预测的时间是 {}\nSvm回归预测的时间是 {}'.format(dt_pre, knn_pre,
    #                                                                                                  rf_pre, svm_pre))

    # 图像预处理
    # import cv2
    # import numpy as np
    #
    # import utils.utils as ut
    #
    # img_9 = cv2.imread('../sources/dataset/test/9-3-410.jpg', 0)
    # img_0 = cv2.imread('../sources/dataset/test/0-1-003.jpg', 0)
    # img_3 = cv2.imread('../sources/dataset/test/3-5-190.jpg', 0)
    # img_6 = cv2.imread('../sources/dataset/test/6-7-402.jpg', 0)
    # img_12 = cv2.imread('../sources/dataset/test/12-14-135.jpg', 0)
    #
    # ret0 = ut.standardization(img_0)
    # ret3 = ut.standardization(img_3)
    # ret6 = ut.standardization(img_6)
    # ret9 = ut.standardization(img_9)
    # ret12 = ut.standardization(img_12)
    #
    # region0, afterDenoise0 = ut.find_max_region(ret0, 15, denoiseScheme='default')
    # region3, afterDenoise3 = ut.find_max_region(ret3, 15, denoiseScheme='default')
    # region6, afterDenoise6 = ut.find_max_region(ret6, 15, denoiseScheme='default')
    # region9, afterDenoise9 = ut.find_max_region(ret9, 15, denoiseScheme='default')
    # region12, afterDenoise12 = ut.find_max_region(ret12, 15, denoiseScheme='default')
    #
    # y0, _0 = ut.surfaceFitting(ret0, deg=2, mbSize=15)
    # fitted_location_img0 = np.copy(img_0)
    # original_location_img0 = np.copy(img_0)
    # for i in range(fitted_location_img0.shape[1]):
    #     fitted_location_img0[y0[i]][i] = 255
    # for i in range(len(_0)):
    #     k = _0[i]
    #     original_location_img0[k[0]][k[1] - 1] = 255
    # flattened_img0 = ut.flatten(img_0, [512 - i for i in y0])
    #
    # y3, _3 = ut.surfaceFitting(ret3, deg=2, mbSize=15)
    # fitted_location_img3 = np.copy(img_3)
    # original_location_img3 = np.copy(img_3)
    # for i in range(fitted_location_img3.shape[1]):
    #     fitted_location_img3[y3[i]][i] = 255
    # for i in range(len(_3)):
    #     k = _3[i]
    #     original_location_img3[k[0]][k[1] - 1] = 255
    # flattened_img3 = ut.flatten(img_3, [512 - i for i in y3])
    #
    # y6, _6 = ut.surfaceFitting(ret6, deg=2, mbSize=15)
    # fitted_location_img6 = np.copy(img_6)
    # original_location_img6 = np.copy(img_6)
    # for i in range(fitted_location_img6.shape[1]):
    #     fitted_location_img6[y6[i]][i] = 255
    # for i in range(len(_6)):
    #     k = _6[i]
    #     original_location_img6[k[0]][k[1] - 1] = 255
    # flattened_img6 = ut.flatten(img_6, [512 - i for i in y6])
    #
    # y9, _9 = ut.surfaceFitting(ret9, deg=2, mbSize=15)
    # fitted_location_img9 = np.copy(img_9)
    # original_location_img9 = np.copy(img_9)
    # for i in range(fitted_location_img9.shape[1]):
    #     fitted_location_img9[y9[i]][i] = 255
    # for i in range(len(_9)):
    #     k = _9[i]
    #     original_location_img9[k[0]][k[1] - 1] = 255
    # flattened_img9 = ut.flatten(img_9, [512 - i for i in y9])
    #
    # y12, _12 = ut.surfaceFitting(ret12, deg=3, mbSize=15)
    # fitted_location_img12 = np.copy(img_12)
    # original_location_img12 = np.copy(img_12)
    # for i in range(fitted_location_img12.shape[1]):
    #     fitted_location_img12[y12[i]][i] = 255
    # for i in range(len(_12)):
    #     k = _12[i]
    #     original_location_img12[k[0]][k[1] - 1] = 255
    # flattened_img12 = ut.flatten(img_12, [512 - i for i in y12])
    #
    # # merge = np.hstack((img, ret, afterDenoise, region, original_location_img, fitted_location_img, flattened_img))
    # merge1 = np.hstack((original_location_img0, original_location_img3, original_location_img6, original_location_img9,
    #                     original_location_img12))
    # merge2 = np.hstack(
    #     (fitted_location_img0, fitted_location_img3, fitted_location_img6, fitted_location_img9, fitted_location_img12))
    # merge3 = np.hstack(
    #     (flattened_img0, flattened_img3, flattened_img6, flattened_img9, flattened_img12))
    #
    # cv2.imwrite('original_location.jpg', merge1)
    # cv2.imwrite('fitted_location.jpg', merge2)
    # cv2.imwrite('flattened_img.jpg', merge3)
