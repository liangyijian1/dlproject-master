import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

from net.MyDataset import *
from net.octnet import octnet
from net.resnet18 import init_weight, ResNet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
writer = SummaryWriter(log_dir='scalar/train')


def startTrain(net, trainLoader, testLoader, valLoader, epoch, lossFun, optimizer, scheduler, trainLog, testLog, valLog,
               savePath=None):
    for k in range(epoch):
        print('\nEpoch: %d' % (k + 1))
        net.train()
        sum_loss = 0.0
        total = 0
        for i, trainData in enumerate(trainLoader):
            length = len(trainLoader)
            total = length
            inputs, labels = trainData
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = lossFun(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            writer.add_scalar('Train', sum_loss / (i + 1), (i + 1 + k * length))
            print('[tra epoch:%d, iter:%d] |Loss: %.05f' % (k + 1, (i + 1 + k * length), sum_loss / (i + 1)))
        writer.add_scalar('scalar/train', sum_loss / total, epoch)
        print('[tra epoch:{}] | Average Loss：{:.5f}'.format(k + 1, sum_loss / total))
        trainLog.write('[tra epoch:{}] | Average Loss：{:.5f}'.format(k + 1, sum_loss / total))
        trainLog.write('\n\n')
        trainLog.flush()
        if savePath is not None:
            print('Saving model......')
            torch.save(net.state_dict(), savePath + 'net_{}.pth'.format(k + 1))
        print("Waiting Val!")
        sum_loss = 0.0
        total = 0
        with torch.no_grad():
            for i, valData in enumerate(valLoader):
                length = len(valLoader)
                total = length
                inputs, labels = valData
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = lossFun(outputs, labels)
                sum_loss += loss.item()
                print('[val epoch:%d, iter:%d] |Loss: %.05f' % (k + 1, (i + 1 + k * length), sum_loss / (i + 1)))
                writer.add_scalar('scalar/val', sum_loss / (i + 1), (i + 1 + k * length))
            print('[val epoch:{}] | Average Loss：{:.5f}'.format(k + 1, sum_loss / total))
            valLog.write('[val epoch:{}] | Average Loss：{:.5f}'.format(k + 1, sum_loss / total))
            valLog.write('\n')
            valLog.flush()
        scheduler.step(sum_loss)
    print("\nWaiting test!")
    sum_loss = 0.0
    total = 0
    with torch.no_grad():
        for i, testData in enumerate(testLoader):
            total = len(testLoader)
            inputs, labels = testData
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = lossFun(outputs, labels)
            sum_loss += loss.item()
            print('Loss: %.03f' % (sum_loss / (i + 1)))
            writer.add_scalar('scalar/test', sum_loss / (i + 1), i)
            testLog.write('learning rate: {}'.format(scheduler.optimizer.defaults['lr']))
            testLog.write('\n')
            testLog.write('Loss: %.03f' % (sum_loss / (i + 1)))
            testLog.write('\n')
            testLog.flush()
        print('Average Loss：{:.5f}'.format(sum_loss / total))
        testLog.write('Average Loss：{:.5f}'.format(sum_loss / total))
        testLog.write('\n')
        testLog.flush()


EPOCH = 50
BATCH_SIZE = 5
LR = 0.01
imgPath = '../sources/dataset/afterFlatten/roi_nlm/'
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(256, 256)),
    torchvision.transforms.ToTensor(),
])

full_dataset = torchvision.datasets.ImageFolder(imgPath, transform_train)
with open('./label.txt', 'w+') as f:
    json.dump(full_dataset.class_to_idx, f)

trainLoader, testLoader, valLoader = load_local_dataset(full_dataset, BATCH_SIZE)
net = octnet(5).to(device)
# net = ResNet18(5).to(device)
# # net = ResNet50(ResidualBlock, 1, 5).to(device)
# # preDict = torch.load('./model/preTrainModel/net_14.pth')
# # preDict.pop('fc.weight')
# # preDict.pop('fc.bias')
# # net.load_state_dict(preDict, strict=False)
# net.apply(init_weight)
# images = torch.randn(1, 1, 28, 28)
# writer.add_graph(net, images)
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=0.001, momentum=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)

if __name__ == "__main__":
    modelPath = './model/'
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    if not os.path.exists('./res'):
        os.makedirs('./res')
    with open("res/test.txt", "w") as f:
        with open("res/log.txt", "w") as f2:
            with open("res/val.txt", 'w') as f3:
                startTrain(net, trainLoader, testLoader, valLoader,
                           EPOCH, loss, optimizer, scheduler, f2, f, f3, savePath=modelPath)
    writer.close()
