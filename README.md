# ResNet18 Training and Adversarial Attacks

This repository provides scripts for training a ResNet18 model on the CIFAR-10 dataset using different training strategies. The following scripts are included:

1. **Standard Training**: Basic training of ResNet18.
2. **PGD Untargeted Training**: Training with Projected Gradient Descent (PGD) adversarial attacks.
3. **PGD Targeted Training**: Training with PGD for targeted adversarial attacks.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- logging

Install the required packages using pip:

```bash
pip install torch torchvision
```
## Dataset
The CIFAR-10 dataset is used for training and testing. Ensure the dataset is available in the ./data directory. It can be downloaded using torchvision.datasets.

## Scripts

1) Train a Resnet on CIFAR 10 using standard training and PGD training and Analyze the class-distribution of misclassification using untargeted attacks.
2) Perform PGD training with targeted adversarial training using adversarial examples with all target classes except ground truth for each sample.
3) Repeat targeted PGD training considering different norms of robust losses across classes instead of averaging.
4) Analyze class-wise natural and robust accuracies in each scenario.
   
### 1. Standard Training
Filename: train_standard.py
Description: Trains a ResNet18 model on CIFAR-10 using standard training procedures.
Logging: Progress is logged to std_train_output.log.
Model Saving: The trained model is saved as standard_resnet18.pth.
### 2. PGD Training
Filename: train_pgd.py
Description: Trains ResNet18 on CIFAR-10 with Projected Gradient Descent (PGD) adversarial attacks.
Logging: Progress is logged to pdg_train_output.log.
Model Saving: The trained model is saved as npgd_resnet18.pth.
### 3. PGD Targeted Training
Filename: train_pgd_targeted.py
Description: Trains ResNet18 on CIFAR-10 with PGD targeted adversarial attacks.
Logging: Progress is logged to pdg_Tar_train_output.log.
Model Saving: The trained model is saved as pgd_Targated_resnet18.pth.

## Results:
**Standard training accuracy** = 0.999 (100 epochs),
**Standard test accuracy** = 0.930,

**PGD_Untargeted_training accuracy** = 0.649 (200 epochs), 
**PGD Test accuracy on adverisl images** = 0.498, 
**PGD test accuracy on orginal images** = 0.8136, 

**PGD_targeted_training accuracy** = 0.649 (200 epochs), 
**PGD Test accuracy on adverisl images** = 0.384, 
**PGD test accuracy on orginal images** = 0.838, 

## Usage
To run any of the scripts, simply execute them with Python:
```bash
python train_standard.py
python train_pgd.py
python train_pgd_targeted.py
```
