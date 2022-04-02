import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

from net.MyDataset import *
from net.resnet18 import init_weight
from net.resnet50 import ResNet50, ResidualBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('./Result')


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
        writer.add_scalar('Train Ave', sum_loss / total, epoch)
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
            writer.add_scalar('Val', sum_loss / total, epoch)
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
            testLog.write('learning rate: {}'.format(scheduler.optimizer.defaults['lr']))
            testLog.write('\n')
            testLog.write('Loss: %.03f' % (sum_loss / (i + 1)))
            testLog.write('\n')
            testLog.flush()
        print('Average Loss：{:.5f}'.format(sum_loss / total))
        testLog.write('Average Loss：{:.5f}'.format(sum_loss / total))
        testLog.write('\n')
        testLog.flush()


EPOCH = 300
BATCH_SIZE = 10
LR = 0.01
imgPath = '../sources/pre_train/'
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
])

full_dataset = torchvision.datasets.ImageFolder(imgPath, './label.txt', transform_train)

trainLoader, testLoader, valLoader = load_local_dataset(full_dataset, BATCH_SIZE)
net = ResNet50(ResidualBlock, 1, 4).to(device)
net.apply(init_weight)

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)

if __name__ == "__main__":
    modelPath = './model/'
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    if not os.path.exists('./res'):
        os.makedirs('./res')
    with open("res/preTrainRes/test.txt", "w") as f:
        with open("res/preTrainRes/log.txt", "w") as f2:
            with open("res/preTrainRes/val.txt", 'w') as f3:
                startTrain(net, trainLoader, testLoader, valLoader,
                           EPOCH, loss, optimizer, scheduler, f2, f, f3, savePath=modelPath)
