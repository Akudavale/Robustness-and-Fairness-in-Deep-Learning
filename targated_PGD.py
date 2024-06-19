import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import logging
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models import *
import torch.backends.cudnn as cudnn

logging.basicConfig(filename='pdg_Tar_train_output.log', level=logging.INFO)

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
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False)



net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0002)


scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)


def pgd_targeted_attack(model, images, target_labels, epsilon=0.0314,alpha=0.00784, iters=7):

    images = images.clone().detach().to(device)
    target_labels = target_labels.clone().detach().to(device)
    loss = nn.CrossEntropyLoss()
    
    adv_images = images + torch.zeros_like(images).uniform_(-epsilon, epsilon)
    
    for i in range(iters):
        adv_images.requires_grad_()
        with torch.enable_grad():
            outputs = model(adv_images)
            cost = -loss(outputs, target_labels).to(device) 
        
        grad = torch.autograd.grad(cost, adv_images)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        adv_images = torch.min(torch.max(adv_images, images - epsilon), images + epsilon)
        adv_images = torch.clamp(adv_images, 0, 1).detach()
    
    return adv_images


def train_pgd_targeted(net, trainloader, criterion, optimizer, scheduler, target_label, epochs):
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            target_labels = torch.full_like(labels, target_label) 
            
            adv_inputs = pgd_targeted_attack(net, inputs, target_labels)

            optimizer.zero_grad()
            outputs = net(adv_inputs)
            loss = criterion(outputs, labels)  
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        epoch_acc =  correct / total
        epoch_loss = running_loss / len(trainloader)
        
        logging.info(f'Epoch {epoch + 1} loss: {epoch_loss:.3f}, Accuracy: {epoch_acc:.3f}')


def save_model(net, filename):
    torch.save(net.state_dict(), filename)

def get_testloader():
    return testloader

def get_model():
    return net

if __name__ == "__main__":
    target_label = 0 
    train_pgd_targeted(net, trainloader, criterion, optimizer, scheduler, target_label, epochs=10)
    save_model(net, 'pgd_Targated_resnet18.pth')
