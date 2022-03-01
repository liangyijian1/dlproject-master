import cv2
import numpy as np
import torchvision.transforms

import utils.preprocess as pr
import utils.utils as ut
from net.CnnRegression import ResNet50Regression
from utils.regression import Regression

if __name__ == '__main__':
    reg = Regression()
    X, y = ut.get_data('res/features/done/img.txt')
    # dt, dt_parm, dt_score = reg.dtRegression(X, y, 1, 50, cv_num=5, n_iter=50)
    # print(dt_score)
    a, b, model = reg.mlRegression(X, y)
    # y_pre = a.predict(content.reshape(1, -1))
    # print(y_pre)

    # ut.saveModel('dt_model.pkl', dt)
    # y_pre = dt.predict(content.reshape(1, -1))
    # print(y_pre)
    img = cv2.imread('test/img/16-225.jpg', flags=0)

    pr.getFeatureVectorPlus(img=img,
                         img_save_path='test/img/te-16-225.jpg',
                         res_save_path='test/res/te-16-225.txt',
                         k=190,
                         deg=3,
                         transform=torchvision.transforms.ToTensor(),
                         model=ResNet50Regression(1),
                         model_dict_path='model/net_22.pth',
                         crop=[60, 0, 30, 30]
                         )
    with open('test/res/te-16-225.txt', 'r') as f:
        feature = f.readline().strip().split(',')
        feature = np.array(feature).astype(np.float64)
    y_pre = a.predict(feature.reshape(1, -1))
    print(y_pre)