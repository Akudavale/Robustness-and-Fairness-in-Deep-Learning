import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
from models import *
import torch.backends.cudnn as cudnn

logging.basicConfig(filename='plots.log', level=logging.INFO)


def load_model(model_path, device):
    model = ResNet18()
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model.load_state_dict(torch.load(model_path), )
    model.eval()
    return model

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



def pgd_evaluate_model(net, testloader, device, attack_fn, **attack_params):
    y_pred = []
    y_true = []
    net.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            #images = attack_fn(net, images, labels, **attack_params)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    return y_true, y_pred


def evaluate_pgd_targeted(model, testloader, target_label, device, epsilon=0.0314, alpha=0.00784, iters=7):
    y_pred = []
    y_true = []
    model.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            target_labels = torch.full_like(labels, target_label)  # Set all target labels to the desired target class
            
            adv_images = pgd_targeted_attack(model, images, target_labels, epsilon, alpha, iters)
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    return y_true, y_pred


def pgd_attack(model, images, labels, epsilon=0.0314, alpha=0.00784, iters=7):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    loss = nn.CrossEntropyLoss()
    
    noise = torch.zeros_like(images)
    adv_images = images + noise.uniform_(-epsilon, epsilon)
    
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

def pgd_targeted_attack(model, images, target_labels, epsilon=0.0314,alpha=0.00784, iters=7):

    images = images.clone().detach().to(device)
    target_labels = target_labels.clone().detach().to(device)
    loss = nn.CrossEntropyLoss()
    
    adv_images = images + torch.zeros_like(images).uniform_(-epsilon, epsilon)
    
    for i in range(iters):
        adv_images.requires_grad_()
        with torch.enable_grad():
            outputs = model(adv_images)
            cost = -loss(outputs, target_labels).to(device)  # Negative loss for targeted attack
        
        grad = torch.autograd.grad(cost, adv_images)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        adv_images = torch.min(torch.max(adv_images, images - epsilon), images + epsilon)
        adv_images = torch.clamp(adv_images, 0, 1).detach()
    
    return adv_images

def plot_confusion_matrix(y_true, y_pred, classes, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":

    from standard_training import get_testloader as get_standard_testloader
    from pgd_training import get_testloader as get_pgd_testloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    target_labels = 0

    # Plot confusion matrix for standard model
    standard_testloader = get_standard_testloader()
    standard_model = load_model('standard_resnet18.pth', device)
    y_true_standard, y_pred_standard = evaluate_model(standard_model, standard_testloader, device)
    plot_confusion_matrix(y_true_standard, y_pred_standard, classes, 'confusion_matrix_standard.png')
    accuracy_standard = np.mean(np.array(y_true_standard) == np.array(y_pred_standard))
    logging.info(f'Test accuracy of standard training :{accuracy_standard:.3f} ')

    pgd_testloader = get_pgd_testloader()
    pgd_model = load_model('pgd_resnet18.pth', device)
    y_true_pgd, y_pred_pgd = pgd_evaluate_model(pgd_model, pgd_testloader, device, attack_fn=pgd_attack, epsilon=0.0314, alpha=0.00784, iters=7)
    plot_confusion_matrix(y_true_pgd, y_pred_pgd, classes, 'confusion_matrix_pgd_Untargated.png')
    accuracy_pgd = np.mean(np.array(y_true_pgd) == np.array(y_pred_pgd))
    logging.info(f'Test accuracy of pdg untargated training :{accuracy_pgd:.3f}')
    

    # Plot confusion matrix for Targatted_PGD model
    pgd_testloader = get_pgd_testloader()
    pgd_model = load_model('pgd_Targated_resnet18.pth', device)
    y_true_pgd, y_pred_pgd = evaluate_pgd_targeted(pgd_model, pgd_testloader,target_labels, device)
    plot_confusion_matrix(y_true_pgd, y_pred_pgd, classes, 'confusion_matrix_pgd_Targated.png')
    accuracy_pgd = np.mean(np.array(y_true_pgd) == np.array(y_pred_pgd))
    logging.info(f'Test accuracy of pdg targated training :{accuracy_pgd:.3f}')
 