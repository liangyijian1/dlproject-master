import math
import os
import sys
import traceback

import cv2
import torch.utils.data
import torchvision.transforms
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, \
    median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from net.CnnRegression import mlpRegression
from net.octnet import octnet
from net.resnet18 import ResNet18
from src.utils.regression import Regression

from src.utils.utils import *
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pass
    # train_x, train_y = get_data('./res/vector/trainVector.txt')
    test_x, test_y = get_data('./res/vector/testVector.txt')
    # # scale = StandardScaler()
    # # train_x = scale.fit_transform(train_x)
    # # test_x = scale.fit_transform(test_x)
    # # 转换为张量
    # train_x = torch.from_numpy(train_x)
    # train_y = torch.from_numpy(train_y)
    #
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)
    # # 转换成DataLoader
    # train_data = Data.TensorDataset(train_x, train_y)
    # test_data = Data.TensorDataset(test_x, test_y)
    #
    # train_loader = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    # test_loader = Data.DataLoader(dataset=test_data, batch_size=32, shuffle=True)
    # # mlp训练
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = mlpRegression().double().to(device)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    # loss = nn.MSELoss()
    #
    # startTrain(net, train_loader, test_loader, 20, loss, optimizer, './model/')

    # img = cv2.imread('1-003.jpg', 0)
    # sd = standardization(img)
    # dn = denoise(sd, 15, mbSize=-1)
    # dn1 = denoise(sd, 15, mbSize=3)
    # dn2 = cv2.fastNlMeansDenoising(sd, h=10, templateWindowSize=7, searchWindowSize=21)
    # p1 = cv2.PSNR(img, dn)
    # p2 = cv2.PSNR(img, dn1)
    # p3 = cv2.PSNR(img, dn2)
    # s1 = ssim(img, dn)
    # s2 = ssim(img, dn1)
    # s3 = ssim(img, dn2)
    # print('p1:{:.3f}, p2:{:.3f}, p3:{:.3f}\n'.format(p1, p2, p3))
    # print('s1:{:.3f}, s2:{:.3f}, s3:{:.3f}'.format(s1, s2, s3))
    # hs = np.hstack((img, sd, dn, dn1, dn2))
    # cv2.imwrite('hs.jpg', hs)

    # original = cv2.imread('1-2-373.jpg', 0)
    # flatten = cv2.imread('2-373.jpg', 0)
    # e1 = calc_2D_Entropy(original)
    # e2 = calc_2D_Entropy(flatten)
    # print(e1, ' ', e2)
    # # margin = int((original.shape[1] - int(original.shape[0] / 2)) / 2)
    # # roi1 = extract_ROI(original, margin, -25, winStep=15)
    # # roi2 = extract_ROI(flatten, margin, -25, winStep=15)
    # # ret = np.hstack((roi1[0], roi2[0]))
    # # cv2.imshow('1', ret)
    # # cv2.waitKey(0)

    # lists = os.listdir('../sources/dataset/withoutFlatten/preprocess/')
    # for item in lists:
    #     dataAugmentation('../sources/dataset/withoutFlatten/preprocess/' + item + '/', rotationProbability=35, angle=0)
    #     print(item, ' done!')

    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Resize(size=(256, 256)),
    # ])
    # # img = cv2.imread('3-384.jpg', 0)
    # #
    # # TestModel(ResNet18(5), './model/cnn_model/net_9.pth').testSingleImg(transform(img).view(1, 1, 256, 256), visualization=True, log_dir='scalar/test', comment='test')
    # dataset = torchvision.datasets.ImageFolder('../sources/dataset/afterFlatten/test/', transform)
    # testDataloader = torch.utils.data.DataLoader(dataset, batch_size=15)
    # TestModel(octnet(5), './model/net_12.pth').testDataLoader(testDataloader, True)

    # rootPath = '../sources/dataset/withoutFlatten/test1/'
    # dirs = os.listdir(rootPath)
    # for item in dirs:
    #     imgNames = os.listdir(rootPath + item)
    #     count = 0
    #     for imgName in imgNames:
    #         if imgName[-3:] != 'jpg':
    #             continue
    #         img = cv2.imread(rootPath + item + '/' + imgName, 0)
    #         sd = standardization(img)
    #         denoiseImg = denoise(sd, 15, -1, 30)
    #         if not os.path.exists('../sources/dataset/withoutFlatten/preprocess1/' + item + '/'):
    #             os.mkdir('../sources/dataset/withoutFlatten/preprocess1/' + item + '/')
    #         cv2.imwrite('../sources/dataset/withoutFlatten/preprocess1/' + item + '/' + imgName, denoiseImg)
    #         count += 1
    #         print('class:{}, {} done! Total:{}, left:{}'.format(item, imgName, len(imgNames), len(imgNames) - count))

    # # 手动标注表面展平
    # rootPath = '../sources/dataset/'
    # # imgPathNames = os.listdir(rootPath)
    # imgPathNames = ['temp']
    # for imgPathName in imgPathNames:
    #     imgNames = os.listdir(rootPath + imgPathName + '/')
    #     # imgNames = ['5-129.jpg']
    #     if not os.path.exists(rootPath + imgPathName + '/done/'):
    #         os.mkdir(rootPath + imgPathName + '/done/')
    #     for idx, imgName in enumerate(imgNames):
    #         try:
    #             if imgName[-3:] == 'jpg':
    #                 imgPath = rootPath + imgPathName + '/' + imgName
    #                 img = cv2.imread(imgPath, 0)
    #                 # img = standardization(img)
    #                 loc = MarkLabel('show').markLabel(img, 'temp.jpg')
    #                 y, _ = surfaceFitting(img, manualLoc=loc, deg=3)
    #                 # afterDenoise = denoise(img, 3)
    #                 flattened_img = flatten(img, y)
    #                 # flattened_img = cropImg(flattened_img, 20, 20, 40, 0)
    #                 cv2.imwrite(rootPath + imgPathName + '/done/' + imgName, flattened_img)
    #                 print(imgName, ' 完成。还有{}个'.format(len(imgNames) - idx - 1))
    #         except Exception as e:
    #             with open('./failed', 'a+') as f:
    #                 f.write('\n' + imgName + "处理时候发生错误，： " + e.__str__())
    #             print(imgName + "处理时候发生错误，： " + e.__str__())
    #             traceback.print_exc(file=sys.stdout)
    #
    # # 去噪
    # rootPath = '../sources/dataset/cyclegan/temp/'
    # # imgPathNames = os.listdir(rootPath)
    # imgPathNames = ['waitConvertA']
    # for imgPathName in imgPathNames:
    #     imgNames = os.listdir(rootPath + imgPathName + '/')
    #     # imgNames = ['5-129.jpg']
    #     if not os.path.exists(rootPath + imgPathName + '/done/'):
    #         os.mkdir(rootPath + imgPathName + '/done/')
    #     for idx, imgName in enumerate(imgNames):
    #         try:
    #             if imgName[-3:] == 'jpg':
    #                 imgPath = rootPath + imgPathName + '/' + imgName
    #                 img = cv2.imread(imgPath, 0)
    #                 # img = standardization(img)
    #                 # y, _ = surfaceFitting(img, deg=3, mbSize=15)
    #                 afterDenoise1 = denoise(img, 3, mbSize=-1)
    #                 afterDenoise = cv2.fastNlMeansDenoising(afterDenoise1, h=10, templateWindowSize=7, searchWindowSize=21)
    #                 # flattened_img = flatten(np.copy(img), [512 - i for i in y])
    #                 # flattened_img = cropImg(flattened_img, 20, 20, 40, 0)
    #                 cv2.imwrite(rootPath + imgPathName + '/done/' + imgName, afterDenoise)
    #                 print(imgName, ' 完成。还有{}个'.format(len(imgNames) - idx - 1))
    #         except Exception as e:
    #             with open('./failed', 'a+') as f:
    #                 f.write('\n' + imgName + "处理时候发生错误，： " + e.__str__())
    #             print(imgName + "处理时候发生错误，： " + e.__str__())
    #             traceback.print_exc(file=sys.stdout)

    # rootPath = '../sources/dataset/cyclegan/'
    # imgPathNames = ['trainA']
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToPILImage(),
    #     torchvision.transforms.Resize(size=(256, 256)),
    #     torchvision.transforms.RandomHorizontalFlip(),
    #     torchvision.transforms.ColorJitter(brightness=(0.8, 1.2)),
    #     torchvision.transforms.RandomAffine(0, translate=(0.05, 0.1), scale=(0.8, 1.2)),
    #     torchvision.transforms.GaussianBlur(3),
    #     torchvision.transforms.RandomRotation(25),
    # ])
    # for imgPathName in imgPathNames:
    #     imgNames = os.listdir(rootPath + imgPathName + '/')
    #     k = int(2000 / len(imgNames))
    #     if not os.path.exists(rootPath + imgPathName + '/done/'):
    #         os.mkdir(rootPath + imgPathName + '/done/')
    #     for idx, imgName in enumerate(imgNames):
    #         try:
    #             if imgName[-3:] == 'jpg':
    #                 imgPath = rootPath + imgPathName + '/' + imgName
    #                 img = cv2.imread(imgPath, 0)
    #                 count = 0
    #                 # for i in range(k + 1):
    #                 for i in range(1):
    #                     tr = np.asarray(transform(img))
    #                     cv2.imwrite(rootPath + imgPathName + '/done/' + count.__str__() + '-' + imgName, tr)
    #                     count += 1
    #                 print(imgName, ' 完成。还有{}个'.format(len(imgNames) - idx - 1))
    #         except Exception as e:
    #             print(imgName + "处理时候发生错误，： " + e.__str__())
    #             traceback.print_exc(file=sys.stdout)

    # # 展平
    # y, _ = ut.surfaceFitting(ret0, deg=2, mbSize=15)
    # fitted_location_img0 = np.copy(img_0)
    # original_location_img0 = np.copy(img_0)
    # for i in range(fitted_location_img0.shape[1]):
    #     fitted_location_img0[y0[i]][i] = 255
    # for i in range(len(_0)):
    #     k = _0[i]
    #     original_location_img0[k[0]][k[1] - 1] = 255
    # flattened_img0 = ut.flatten(img_0, [512 - i for i in y0])

    # # 提取ROI
    # rootPath = '../sources/dataset/afterFlatten/test/'
    # # categories = os.listdir(rootPath)
    # categories = ['8', '12', '16', '20']
    # for category in categories:
    #     count = 0
    #     imgNames = os.listdir(rootPath + category + '/')
    #     if not os.path.exists(rootPath + category + '/done/'):
    #         os.mkdir(rootPath + category + '/done/')
    #     for imgName in imgNames:
    #         if not imgName[-3:] == 'jpg':
    #             continue
    #         # if imgName[:7] != 'flip-14':
    #         # if imgName[:1] != '5':
    #         # if imgName[:6] != 'flip-5':
    #         #     continue
    #         try:
    #             img = cv2.imread(rootPath + category + '/' + imgName, flags=0)
    #             margin = int((img.shape[1] - int(img.shape[0] / 2)) / 2)
    #             rets = extract_ROI(img, margin=margin, diff=- int(margin / 2) + 10, winStep=25, k=1)
    #             count += 1
    #             for idx, ret in enumerate(rets):
    #                 cv2.imwrite(rootPath + category + '/done/' + idx.__str__() + '-' + imgName, ret)
    #             print(
    #                 'class: {}, total number: {}, current: {}, left number: {}'.format(category, len(imgNames), imgName,
    #                                                                                    len(imgNames) - count))
    #         except Exception as e:
    #             with open('./failed', 'a+') as f:
    #                 f.write('\n' + imgName + "处理时候发生错误，： " + e.__str__())
    #                 print(imgName + "处理时候发生错误，： " + e.__str__())
    #                 traceback.print_exc(file=sys.stdout)

    # # 获取向量
    # model = octnet(num_class=5, check_fc=False)
    # modelLocation = './model/cnn_model/net_10.pth'
    # rootPath = '../sources/dataset/withoutFlatten/test/'
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Resize(size=(256, 256)),
    # ])
    # getAllFeatureVector(rootPath=rootPath, model=model, modelLocation=modelLocation, transform=transform,
    #                     labelPath='./label.txt')

    # # 机器学习模型训练过程
    # regr = Regression()
    #
    # X, y = get_data('./res/vector/trainVector.txt')
    # model, parm, score = regr.rfRegression(X, y, 1, 200, 1, 30, 1, 20, cv_num=3)
    # # model, parm, score = regr.svmRegression(X, y, [0.01, 0.1, 1.0, 10], [0.01, 0.1, 1.0, 10], [0.01, 0.1, 1.0, 10],
    # #                                         cv_num=3)
    # # 直接使用predict()函数进行预测
    # y_pre = model.predict(X)
    # # 使用utils.py中的saveModel()函数将模型保存到本地
    # saveModel('model/ml_model/rf.pkl', model)
    # print(r2_score(y, y_pre).__str__() + '   ' + score.__str__())

    # # 对testVector验证
    # regr = Regression()
    #
    # X, y = get_data('./res/vector/testVector.txt')
    # rf_pre = loadModel('model/ml_model/rf.pkl').predict(X)
    # # svm_pre = loadModel('model/ml_model/svm.pkl').predict(X)
    # # print('Random Forest预测的r2_score是 {}     Svm预测的r2_score是 {}'.format(r2_score(y, rf_pre), r2_score(y, svm_pre)))
    # print('r2_score是 {:.3f}, MAE:{:.3f}, MedAE:{:.3f}, MSE:{:.3f}, RMSE:{:.3f}'.format(
    #                                                                         r2_score(y, rf_pre),
    #                                                                         mean_absolute_error(y, rf_pre),
    #                                                                         median_absolute_error(y, rf_pre),
    #                                                                         # mean_absolute_percentage_error(y, rf_pre),
    #                                                                         mean_squared_error(y, rf_pre),
    #                                                                         math.sqrt(mean_squared_error(y, rf_pre))))

    # # 使用机器学习模型验证
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Resize(size=(224, 224)),
    # ])
    # img = cv2.imread('../sources/dataset/dataset/12/0-8-435.jpg', 0)
    # feature = TestModel(ResNet50Regression(1), 'model/cnn_model/transfer_resnet50.pth', strict=False) \
    #     .getFeatureVector(transform(img).view(1, 1, 224, 224))
    # rf_pre = loadModel('model/ml_model/svm.pkl').predict(feature.reshape(1, -1))
    # svm_pre = loadModel('model/ml_model/svm.pkl').predict(feature.reshape(1, -1))
    # print('Random Forest回归预测的时间是 {}\nSvm回归预测的时间是 {}'.format(rf_pre, svm_pre))

    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Resize(size=(224, 224)),
    # ])
    # img_9 = cv2.imread('../sources/dataset/test/9-3.jpg', 0)
    # ret9 = standardization(img_9)
    # img_9 = denoise(ret9, 15, 3, 30)
    #
    # img_0 = cv2.imread('../sources/dataset/test/0-1.jpg', 0)
    # ret0 = standardization(img_0)
    # img_0 = denoise(ret0, 15, 3, 30)
    #
    # img_3 = cv2.imread('../sources/dataset/test/3-2.jpg', 0)
    # ret3 = standardization(img_3)
    # img_3 = denoise(ret3, 15, 3, 30)
    #
    # img_6 = cv2.imread('../sources/dataset/test/6-2.jpg', 0)
    # ret6 = standardization(img_6)
    # img_6 = denoise(ret6, 15, 3, 30)
    #
    # img_12 = cv2.imread('../sources/dataset/test/12-3.jpg', 0)
    # ret12 = standardization(img_12)
    # img_12 = denoise(ret12, 15, 3, 30)
    #
    # feature_0 = TestModel(ResNet50Regression(1), 'model/cnn_model/transfer_resnet50.pth', strict=False) \
    #     .getFeatureVector(transform(img_0).view(1, 1, 224, 224))
    # feature_3 = TestModel(ResNet50Regression(1), 'model/cnn_model/transfer_resnet50.pth', strict=False) \
    #     .getFeatureVector(transform(img_3).view(1, 1, 224, 224))
    # feature_6 = TestModel(ResNet50Regression(1), 'model/cnn_model/transfer_resnet50.pth', strict=False) \
    #     .getFeatureVector(transform(img_6).view(1, 1, 224, 224))
    # feature_9 = TestModel(ResNet50Regression(1), 'model/cnn_model/transfer_resnet50.pth', strict=False) \
    #     .getFeatureVector(transform(img_9).view(1, 1, 224, 224))
    # feature_12 = TestModel(ResNet50Regression(1), 'model/cnn_model/transfer_resnet50.pth', strict=False) \
    #     .getFeatureVector(transform(img_12).view(1, 1, 224, 224))
    # rf_pre = loadModel('model/ml_model/svm.pkl').predict(feature_12.reshape(1, -1))
    # svm_pre = loadModel('model/ml_model/svm.pkl').predict(feature_12.reshape(1, -1))
    # print('Random Forest回归预测的时间是 {}\nSvm回归预测的时间是 {}'.format(rf_pre, svm_pre))

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

    # 对文件夹内的图像进行评测
    # rootPath = '../sources/dataset/test/'
    # model = ResNet50Regression(1)
    # modelLocation = './model/cnn_model/transfer_resnet50.pth'
    # transform = torchvision.transforms.ToTensor()
    # getAllFeatureVector(rootPath=rootPath, model=model, modelLocation=modelLocation, transform=transform)
    # make_labels('./res/vector/', save_path='./res/vector/', label_location='./label.txt')

    # # 集合的测试
    # X, y = get_data('./res/trainVector.txt')
    # rf_pre = loadModel('model/ml_model/svm.pkl').predict(X)
    # svm_pre = loadModel('model/ml_model/svm.pkl').predict(X)
    # print(
    #     'Random Forest对测试集的:\nr2_score：{:.4f}， 均方误差MSE:{:.4f}, 绝对均值误差MAE:{:.4f}, 解释方差explained_variance_score:{:.4f}, 绝对中位差median_absolute_error：{:.4f}\n'.format(
    #         r2_score(y, rf_pre), mean_squared_error(y, rf_pre), mean_absolute_error(y, rf_pre),
    #         explained_variance_score(y, rf_pre), median_absolute_error(y, rf_pre)))
    # print(
    #     'SVM对测试集的:\nr2_score：{:.4f}， 均方误差MSE:{:.4f}, 绝对均值误差MAE:{:.4f}, 解释方差explained_variance_score:{:.4f}, 绝对中位差median_absolute_error：{:.4f}\n'.format(
    #         r2_score(y, svm_pre), mean_squared_error(y, svm_pre), mean_absolute_error(y, svm_pre),
    #         explained_variance_score(y, svm_pre), median_absolute_error(y, svm_pre)))
