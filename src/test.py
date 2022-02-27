import torchvision

from net.CnnRegression import ResNet50Regression
from utils.preprocess import getAllFeatureVector

if __name__ == '__main__':
    model = ResNet50Regression(1)
    modelLocation = 'model/net_22.pth'
    rootPath = '../sources/dataset/'
    transform = torchvision.transforms.ToTensor()
    getAllFeatureVector(rootPath=rootPath, model=model, modelLocation=modelLocation, transform=transform)
    # temp = []
    # names = os.listdir(rootPath)
    # for name in names:
    #     imgNames = os.listdir(rootPath + name + '/')
    #     for imgName in imgNames:
    #         img = cv2.imread(rootPath + name + '/' + imgName, flags=0)
    #         transform = torchvision.transforms.ToTensor()
    #         k = TestModel(model=model, modelLocation=modelLocation, strict=False) \
    #             .getFeatureVector(transform(img).view(1, 1, 224, 224))
    #         temp.append(k)
    # print('time:{}\n'.format(time_beg - time_end))
    # print(np.array(temp))
