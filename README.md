1) Train a Resnet on CIFAR 10 using standard training and PGD training and Analyze the class-distribution of misclassification using untargeted attacks.
2) Perform PGD training with targeted adversarial training using adversarial examples with all target classes except ground truth for each sample.
3) Repeat targeted PGD training considering different norms of robust losses across classes instead of averaging.
4) Analyze class-wise natural and robust accuracies in each scenario.

Results:
Standard training accuracy = 0.999 (100 epochs)
Standard test accuracy = 0.930
PGD_training accuracy (Adversial Images + Orginal Images) = 0.943 (200 epochs )
PGD Test accuracy on adverisl images = 0.454 
PGD test accuracy on orginal images = 0.850
