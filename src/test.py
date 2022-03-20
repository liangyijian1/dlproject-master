import cv2
import torchvision

from net.CnnRegression import ResNet50Regression
from src.utils.regression import Regression
from src.utils.utils import loadModel, TestModel

if __name__ == '__main__':
    regr = Regression()

    # X, y = get_data('./res/vector/vector.txt')
    # model, parm, score = regr.rfRegression(X, y, 1, 200, 1, 30, 1, 20, cv_num=10, n_iter=10)
    # # 直接使用predict()函数进行预测
    # y_pre = model.predict(X)
    # # 使用utils.py中的saveModel()函数将模型保存到本地
    # saveModel('model/ml_model/rf.pkl', model)
    # print(r2_score(y, y_pre).__str__() + '   ' + score.__str__())

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=(224, 224)),
    ])
    img_9 = cv2.imread('../sources/dataset/test/9-3.jpg', 0)
    img_0 = cv2.imread('../sources/dataset/test/0-1.jpg', 0)
    img_3 = cv2.imread('../sources/dataset/test/3-2.jpg', 0)
    img_6 = cv2.imread('../sources/dataset/test/6-2.jpg', 0)
    img_12 = cv2.imread('../sources/dataset/test/12-3.jpg', 0)
    feature_0 = TestModel(ResNet50Regression(1), 'model/cnn_model/net_21.pth', strict=False) \
        .getFeatureVector(transform(img_0).view(1, 1, 224, 224))
    feature_3 = TestModel(ResNet50Regression(1), 'model/cnn_model/net_21.pth', strict=False) \
        .getFeatureVector(transform(img_3).view(1, 1, 224, 224))
    feature_6 = TestModel(ResNet50Regression(1), 'model/cnn_model/net_21.pth', strict=False) \
        .getFeatureVector(transform(img_6).view(1, 1, 224, 224))
    feature_9 = TestModel(ResNet50Regression(1), 'model/cnn_model/net_21.pth', strict=False) \
        .getFeatureVector(transform(img_9).view(1, 1, 224, 224))
    feature_12 = TestModel(ResNet50Regression(1), 'model/cnn_model/net_21.pth', strict=False) \
        .getFeatureVector(transform(img_12).view(1, 1, 224, 224))
    dt_pre = loadModel('model/ml_model/dt.pkl').predict(feature_12.reshape(1, -1))
    knn_pre = loadModel('model/ml_model/knn.pkl').predict(feature_12.reshape(1, -1))
    rf_pre = loadModel('model/ml_model/rf.pkl').predict(feature_12.reshape(1, -1))
    svm_pre = loadModel('model/ml_model/svm.pkl').predict(feature_12.reshape(1, -1))
    print('Decison Tree回归预测的时间是 {}\nKnn回归预测的时间是 {}\nRandom Forest回归预测的时间是 {}\nSvm回归预测的时间是 {}'.format(dt_pre, knn_pre,
                                                                                                     rf_pre, svm_pre))

# import cv2
# import numpy as np
#
# import utils.utils as ut
#
# img = cv2.imread('../sources/dataset/12/13-215.jpg', flags=0)
# ret = ut.standardization(img)
# temp = ret
# region, afterDenoise = ut.find_max_region(ret, 5, denoiseScheme='default')
# y, _ = ut.surfaceFitting(ret, deg=1, mbSize=5)
# fitted_location_img = np.copy(img)
# original_location_img = np.copy(img)
# for i in range(fitted_location_img.shape[1]):
#     fitted_location_img[y[i]][i] = 255
# for i in range(len(_)):
#     k = _[i]
#     original_location_img[k[0]][k[1] - 1] = 255
# flattened_img = ut.flatten(img, [512 - i for i in y])
# merge = np.hstack((img, ret, afterDenoise, region, original_location_img, fitted_location_img, flattened_img))
# cv2.imwrite('0-1-051-.jpg', merge)
