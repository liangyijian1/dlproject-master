import torch.nn as nn
import torch.optim as optim

from MyDataset import *
from net.resnet18 import init_weight
from net.resnet50 import ResNet50, ResidualBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def startTrain(net, trainLoader, testLoader, valLoader, epoch, lossFun, optimizer, trainLog, testLog, valLog,
               savePath=None):
    for k in range(epoch):
        print('\nEpoch: %d' % (k + 1))
        net.train()
        sum_loss = 0.0
        for i, trainData in enumerate(trainLoader):
            length = len(trainLoader)
            inputs, labels = trainData
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = lossFun(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            print('[tra epoch:%d, iter:%d] |Loss: %.03f' % (k + 1, (i + 1 + k * length), sum_loss / (i + 1)))
            trainLog.write('%03d  %05d |Loss: %.03f' % (k + 1, (i + 1 + k * length), sum_loss / (i + 1)))
            trainLog.write('\n')
            trainLog.flush()
        if savePath is not None:
            print('Saving model......')
            torch.save(net.state_dict(), savePath + 'net_{}.pth'.format(k + 1))
        print("Waiting Val!")
        sum_loss = 0.0
        with torch.no_grad():
            for i, valData in enumerate(valLoader):
                length = len(valLoader)
                inputs, labels = valData
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = lossFun(outputs, labels)
                sum_loss += loss.item()
                print('[val epoch:%d, iter:%d] |Loss: %.03f' % (k + 1, (i + 1 + k * length), sum_loss / (i + 1)))
                valLog.write('%03d  %05d |Loss: %.03f' % (k + 1, (i + 1 + k * length), sum_loss / (i + 1)))
                valLog.write('\n')
                valLog.flush()
        scheduler.step(sum_loss)
    print("\nWaiting test!")
    sum_loss = 0.0
    with torch.no_grad():
        for i, testData in enumerate(testLoader):
            inputs, labels = testData
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = lossFun(outputs, labels)
            sum_loss += loss.item()
            print('Loss: %.03f' % (sum_loss / (i + 1)))
            testLog.write('Loss: %.03f' % (sum_loss / (i + 1)))
            testLog.write('\n')
            testLog.flush()


EPOCH = 300
BATCH_SIZE = 5
LR = 0.01

imgPath = '../sources/dataset/'
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation((-20, 20)),
    torchvision.transforms.ToTensor(),
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
full_dataset = torchvision.datasets.ImageFolder(imgPath, 'label.txt', transform_train)

trainloader, testloader, valloader = load_local_dataset(full_dataset, BATCH_SIZE)
net = ResNet50(ResidualBlock, 1, 3).to(device)
net.apply(init_weight)

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

if __name__ == "__main__":
    modelPath = '../../autodl-nas/model/'
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    if not os.path.exists('history/res'):
        os.makedirs('history/res')
    with open("history/res/test.txt", "w") as f:
        with open("history/res/log.txt", "w") as f2:
            with open("history/res/val.txt", 'w') as f3:
                startTrain(net, trainloader, testloader, valloader,
                           EPOCH, loss, optimizer, f2, f, f3, savePath=modelPath)
