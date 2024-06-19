import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging
from torch.optim import lr_scheduler
from models import *
import torch.backends.cudnn as cudnn

logging.basicConfig(filename='std_train_output.log', level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(device)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

data_path = './data'
trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=100, shuffle=False)


net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True


"""
net = resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 10)
net = net.to(device)
"""

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0002)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

def train_standard(net, trainloader, criterion, optimizer, epochs):
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        scheduler.step()
        
        epoch_acc = correct / total
        epoch_loss = running_loss / len(trainloader)
        
        logging.info(f'Epoch {epoch + 1} loss: {epoch_loss:.3f} Accuracy: {epoch_acc:.3f}')

# For use in the confusion matrix plotting script
def get_testloader():
    return testloader

def get_model():
    return net

if __name__ == "__main__":
    train_standard(net, trainloader, criterion, optimizer, epochs=100)
    torch.save(net.state_dict(), 'standard_resnet18.pth')
    logging.info("Model saved")

