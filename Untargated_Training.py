import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, models
from collections import defaultdict
import numpy as np
import random
from torch.optim import lr_scheduler
from models import *
import torch.backends.cudnn as cudnn
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

logging.basicConfig(filename='PGD_Untargated_Average_Loss_Resnet18.log', level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(device)


model= ResNet18()
model = model.to(device)
model = torch.nn.DataParallel(model)
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss(reduction='none')

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0002)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, images_per_class):
        self.dataset = dataset
        self.images_per_class = images_per_class

        # Store indices of each class
        self.class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset):
            self.class_indices[label].append(idx)

        # Calculate the number of batches
        self.num_classes = len(self.class_indices)
        self.num_batches = min(len(indices) // images_per_class for indices in self.class_indices.values())
        self.batch_indices = self.generate_balanced_batches()

    def generate_balanced_batches(self):
        batch_indices = []
        for _ in range(self.num_batches):
            batch = []
            for class_indices in self.class_indices.values():
                batch.extend(np.random.choice(class_indices, self.images_per_class, replace=False))
            np.random.shuffle(batch)
            batch_indices.append(batch)
        return batch_indices

    def __len__(self):
        return len(self.batch_indices)

    def __iter__(self):
        for batch in self.batch_indices:
            yield batch

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

def Train( org_loss, loss_class, l2, net, trainloader, optimizer, scheduler, epochs,):
    net.train()
    loss_data = []
    accuracy = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss(reduction='none')
    num_classes = 10

    if loss_class:
        logging.info('Whole_loss is False')
        org_loss=False

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            adv_inputs = pgd_untargated_attack(net, inputs, labels)

            optimizer.zero_grad()

            outputs = net(adv_inputs)

            if org_loss:
                output2 = net(inputs)
                loss = criterion(outputs, labels) + criterion(output2, labels)
            else:
                loss = criterion(outputs, labels) 
            
            

            if loss_class:
                loss_per_class = torch.zeros(num_classes).to(device)
                for j in range(num_classes):
                    class_req = (labels == j)
                    class_loss = loss[class_req]
                    if l2:
                        class_norm = torch.norm(class_loss, p=2)      # L2 norm
                    else:
                        class_norm = torch.sqrt(torch.sum(class_loss))  # SSQE (L0.5 norm)
                    loss_per_class[j] += class_norm

                mean_norm = loss_per_class.mean()
                mean_norm.backward()

            else:
                mean_norm = torch.mean(loss)
                mean_norm.backward()
                
            optimizer.step()

            running_loss += mean_norm.item()
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += torch.eq(predicted, labels).sum().item()

        scheduler.step()

        epoch_acc = correct / total
        epoch_loss = running_loss / len(trainloader)
        loss_data.append(epoch_loss)
        accuracy.append(epoch_acc)
        logging.info(f'Epoch {epoch + 1} loss: {epoch_loss:.3f}, Accuracy: {epoch_acc:.3f}')

    return loss_data, accuracy

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

def plot_confusion_matrix2(true_labels, predicted_labels, classes,title, filename):
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

def plot_confusion_matrix(true_labels, predicted_labels, classes, title, filename):
    cm = confusion_matrix(true_labels, predicted_labels)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)
    plt.figure(figsize=(15, 10))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def load_model(model_path, device):
    model = ResNet18()
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model.load_state_dict(torch.load(model_path), )
    model.eval()
    return model

def plot_loss_vs_epoch(all_loss_data,accuracy , filename, title):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(all_loss_data)+1), all_loss_data, label='Loss', color='blue')
    plt.plot(range(1, len(accuracy)+1), accuracy, label='Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":

    num_epochs = 2
    Balanced = False # True then balanced batch loader (Equal no of samples per batch)
    org_loss = False # True when you want to calculate adversarial + orginal images loss and perform bacwardpass.(Generally we keptKeep false if loss class is true)
    loss_class = False #True if you want to calculate class wise loss.
    l2 = False # True only valid when loss_class is True, and if loos_class is True and L2 is false then it L0.5(sqrt) norm.

    if Balanced:
        logging.info("Balanced Batch")
        # With batch sampler 
        batch_size = 90 
        images_per_class = 9
        sampler = BalancedBatchSampler(train_dataset, images_per_class)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, shuffle=False)
    else:
        logging.info("Un_Balanced Batch")
        #without batch sampler 
        batch_size = 100 
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    

    # Below all the file name should be changed manually according to training procedure, just the type of loss (Average, L2 or L0.5) rest all remains same
    #Training
    train_loss, train_acc  = Train(org_loss,  loss_class, l2 , net=model, trainloader=train_loader, optimizer=optimizer,scheduler= scheduler,epochs= num_epochs)
    torch.save(model.state_dict(), 'PGD_Untargated_Average_Loss_Resnet18.pth')
    plot_loss_vs_epoch(train_loss, train_acc ,filename= "PGD_Untargated_Average_Loss_Training_Loss_&_Accuracy_vs_Epoch.png", title="PGD_Untargated_attack_Loss_Training_Loss_&_Accuracy_vs_Epoch")
    


    #evaluating
    batch_size = 90
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #Targated_model = model
    Targated_model = load_model('PGD_Untargated_Average_Loss_Resnet18.pth', device)
    classes = train_dataset.classes

    #standard eval
    test_loss, test_acc, all_preds, all_targets = std_test(Targated_model, device, test_loader, criterion)
    logging.info(f'Final Test Loss: {test_loss:.3f}, Final Test Accuracy: {test_acc:.4f} with Standard_test_data')
    plot_confusion_matrix(all_targets, all_preds, classes, title= "PGD_Untargated_Average_Loss_Confusion_matrix_Standard_Data", filename="PGD_Untargated_Average_Loss_Confusion_matrix_Standard_Data.png")

    #untargated eval
    test_loss, test_acc, all_preds, all_targets = pgd_untargated_test(Targated_model, device, test_loader, criterion)
    logging.info(f'Final Test Loss: {test_loss:.3f}, Final Test Accuracy: {test_acc:.4f} with PGD_untargated_test_data')
    plot_confusion_matrix(all_targets, all_preds, classes,title= "PGD_Untargated_Average_Loss_Confusion_matrix_Untargated_Data", filename="PGD_Untargated_Average_Loss_Confusion_matrix_Untargated_Data.png")

    #Targated eval
    test_loss, test_acc, all_preds, all_targets = pgd_targated_test(Targated_model, device, test_loader, criterion)
    logging.info(f'Final Test Loss: {test_loss:.3f}, Final Test Accuracy: {test_acc:.4f} with PGD_targated_test_data')
    plot_confusion_matrix(all_targets, all_preds, classes,title= "PGD_Untargated_Average_Loss_Confusion_matrix_Targated_Data",  filename="PGD_Untargated_Average_Loss_Confusion_matrix_Targated_Data.png")



