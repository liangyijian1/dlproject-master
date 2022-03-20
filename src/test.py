# import cv2
# import torchvision
# from sklearn.metrics import r2_score
#
# from net.CnnRegression import ResNet50Regression
# from src.utils.preprocess import TestModel
# from src.utils.regression import Regression
# from src.utils.utils import loadModel, get_data, saveModel
#
# if __name__ == '__main__':
#     regr = Regression()
#
#     # X, y = get_data('./res/vector/vector.txt')
#     # model, parm, score = regr.knnRegression(X, y, 1, 30, cv_num=10)
#     # # 直接使用predict()函数进行预测
#     # y_pre = model.predict(X)
#     # # 使用utils.py中的saveModel()函数将模型保存到本地
#     # saveModel('./model/ml_model/knn.pkl', model)
#     # print(r2_score(y, y_pre).__str__() + '   ' + score.__str__())
#
#     transform = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Resize(size=(224, 224)),
#     ])
#     img = cv2.imread('../sources/dataset/test/9-1.jpg', 0)
#     feature = TestModel(ResNet50Regression(1), 'model/cnn_model/net_21.pth', strict=False) \
#         .getFeatureVector(transform(img).view(1, 1, 224, 224))
#     y_pre = loadModel('model/ml_model/knn.pkl').predict(feature.reshape(1, -1))
#     print(y_pre)

import cv2
import numpy as np

import utils.utils as ut

img = cv2.imread('../sources/dataset/12/13-215.jpg', flags=0)
ret = ut.standardization(img)
temp = ret
region, afterDenoise = ut.find_max_region(ret, 5, denoiseScheme='default')
y, _ = ut.surfaceFitting(ret, deg=1, mbSize=5)
fitted_location_img = np.copy(img)
original_location_img = np.copy(img)
for i in range(fitted_location_img.shape[1]):
    fitted_location_img[y[i]][i] = 255
for i in range(len(_)):
    k = _[i]
    original_location_img[k[0]][k[1] - 1] = 255
flattened_img = ut.flatten(img, [512 - i for i in y])
merge = np.hstack((img, ret, afterDenoise, region, original_location_img, fitted_location_img, flattened_img))
cv2.imwrite('0-1-051-.jpg', merge)
