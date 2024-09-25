# One-Shot Machine Unlearning with Mnemonic Code

Code for One-Shot Machine Unlearning with Mnemonic Code

## Environment
```
torch==2.0.0+cu117
torchvision==0.15.1+cu117
```

## Argument
--conf: Path to the configuration file for experiment.

## Arguments in conf file
```
--seed: Random seed for the experiment.
--dataname: Using data (mnist, cifar10, cub, stn, imagenet).
--num_classes: Class size of the dataset.
--t_mix: The probability of replacing the training data with the mnemonic codes
--model: Target model (mlp, resnet18, pretrained-resnet18).
--train_batch: Batch size for training.
--opt: Optimizer for training.
--epochs: Epochs for training.
--lr: Learning rate for training.
--weight_decay: The size of weight decay.
--scheduler: Scheduler for training.
--forget_idx: The forgetting class.
--lambda1: $\lambda_1$ in main paper.
--lambda2: $\lambda_2$ in main paper.
```

## Command
Experiment for MNIST (Download for MNIST will begin in ./data).
```
python3 main.py --conf conf/mnist.yaml
```

Experiment for CIFAR10 (Download for CIFAR10 will begin in ./data).
```
python3 main.py --conf conf/cifar.yaml
```

Experiment for CUB (You should put the CUB dataset in ./data).
```
python3 main.py --conf conf/cub.yaml
```

Experiment for STN (You should put the STN dataset in ./data).
```
python3 main.py --conf conf/stn.yaml
```
