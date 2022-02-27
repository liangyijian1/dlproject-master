from main import testloader
from net.CnnRegression import ResNet50Regression
from utils.preprocess import TestModel

if __name__ == '__main__':
    model = ResNet50Regression(1)
    TestModel(model=model, modelLocation='model/net_22.pth', strict=False).getFeatureVector(testloader)

