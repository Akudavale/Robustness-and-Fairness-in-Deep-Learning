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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import random
logging.basicConfig(filename='Standard3_Resnet18.log', level=logging.INFO)

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


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0002)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

def generate_targeted_labels(true_labels, num_classes=10):
    targeted_labels = true_labels.clone().detach()
    for i in range(len(true_labels)):
        possible_targets = list(range(num_classes))
        possible_targets.remove(true_labels[i].item())
        targeted_labels[i] = random.choice(possible_targets)
    return targeted_labels

def pgd_untargated_attack(model, images, labels):

    epsilon=0.0314 
    alpha=0.00784 
    iters=7
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    loss = nn.CrossEntropyLoss()
    

    adv_images = images + torch.zeros_like(images).uniform_(-epsilon, epsilon)
    
    for i in range(iters):
        adv_images.requires_grad_()
        with torch.enable_grad():
            outputs = model(adv_images)
            cost = loss(outputs, labels).to(device)
        
        grad = torch.autograd.grad(cost, adv_images)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        adv_images = torch.min(torch.max(adv_images, images - epsilon), images + epsilon)
        adv_images = torch.clamp(adv_images, 0, 1).detach()
    
    return adv_images

def pgd_targated_attack(model, images,  target_labels):
    
    epsilon=0.0314 
    alpha=0.00784 
    iters=7

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def train_standard(net, trainloader, criterion, optimizer, epochs):
    net.train()
    loss_data =[]
    accuracy =[]

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

        loss_data.append(epoch_loss)
        accuracy.append(epoch_acc)
        logging.info(f'Epoch {epoch + 1} loss: {epoch_loss:.3f} Accuracy: {epoch_acc:.3f}')


    return loss_data, accuracy


def plot_confusion_matrix(true_labels, predicted_labels, classes,title, filename):
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(15,10))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def evaluate_model(net, testloader, device):
    y_pred = []
    y_true = []
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    return y_true, y_pred

def plot_loss_vs_epoch(all_loss_data,accuracy, filename, title):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(all_loss_data)+1), all_loss_data, label='Loss', color='blue')
    plt.plot(range(1, len(accuracy)+1), accuracy, label='Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def load_model(model_path, device):
    model = ResNet18()
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model.load_state_dict(torch.load(model_path), )
    model.eval()
    return model

def std_test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy =  correct / total

    return test_loss, accuracy, all_preds, all_targets

def pgd_untargated_test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            perturbed_data = pgd_untargated_attack(model, data, target)

            output = model(perturbed_data)
            test_loss += criterion(output, target).item()

            _, predicted = torch.max(output, dim=1)
            total += target.size(0)
            correct += torch.eq(predicted,target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy =  correct / total

    return test_loss, accuracy, all_preds, all_targets

def pgd_targated_test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            target_labels = generate_targeted_labels(target)
            perturbed_data = pgd_targated_attack(model, data, target_labels)

            output = model(perturbed_data)
            test_loss += criterion(output, target).item()

            _, predicted = torch.max(output, dim=1)
            total += target.size(0)
            correct += torch.eq(predicted,target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy =  correct / total

    return test_loss, accuracy, all_preds, all_targets
if __name__ == "__main__":

    epochs = 60

    #Training
    loss_data, accuracy = train_standard(net, trainloader, criterion, optimizer, epochs)
    torch.save(net.state_dict(), 'Standard_Resnet18.pth')
    plot_loss_vs_epoch(loss_data, accuracy, filename= "Standard_Training_Loss_&_Accuracy_vs_Epoch.png", title="Standard_Training_Loss_&_Accuracy_vs_Epoch")
    
    classes = trainset.classes
    standard_model = load_model('Standard_Resnet18.pth', device)

    #standard eval
    test_loss, test_acc, all_preds, all_targets = std_test(standard_model, device, testloader, criterion)
    logging.info(f'Final Test Loss: {test_loss:.3f}, Final Test Accuracy: {test_acc:.4f} with Standard_test_data')
    plot_confusion_matrix(all_targets, all_preds, classes, title= "PGD_Untargated_Average_Loss_Confusion_matrix_Standard_Data", filename="PGD_Untargated_Average_Loss_Confusion_matrix_Standard_Data.png")

    #untargated eval
    test_loss, test_acc, all_preds, all_targets = pgd_untargated_test(standard_model, device, testloader, criterion)
    logging.info(f'Final Test Loss: {test_loss:.3f}, Final Test Accuracy: {test_acc:.4f} with PGD_untargated_test_data')
    plot_confusion_matrix(all_targets, all_preds, classes,title= "PGD_Untargated_Average_Loss_Confusion_matrix_Untargated_Data", filename="PGD_Untargated_Average_Loss_Confusion_matrix_Untargated_Data.png")

    #Targated eval
    test_loss, test_acc, all_preds, all_targets = pgd_targated_test(standard_model, device, testloader, criterion)
    logging.info(f'Final Test Loss: {test_loss:.3f}, Final Test Accuracy: {test_acc:.4f} with PGD_targated_test_data')
    plot_confusion_matrix(all_targets, all_preds, classes,title= "PGD_Untargated_Average_Loss_Confusion_matrix_Targated_Data",  filename="PGD_Untargated_Average_Loss_Confusion_matrix_Targated_Data.png")
    