import torch
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt


def load_MNIST(root_path='.', download=False, transform=None):
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    mnist_data_train = torchvision.datasets.MNIST(root_path, train=True, download=download, transform=transform)
    mnist_data_test = torchvision.datasets.MNIST(root_path, train=False, download=False, transform=transform)
    return mnist_data_train, mnist_data_test


def load_CIFAR(root_path='./CIFAR-10', cls: int = 10, download=False, transform=None):
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    if cls == 10:
        CIFAR = torchvision.datasets.CIFAR10
    elif cls == 100:
        CIFAR = torchvision.datasets.CIFAR100
    cifar_data_train = CIFAR(root_path, train=True, download=download, transform=transform)
    cifar_data_test = CIFAR(root_path, train=False, download=False, transform=transform)
    return cifar_data_train, cifar_data_test


if __name__ == '__main__':
    # train, test = load_MNIST(download=False)
    train, test = load_CIFAR(download=False)
    pic_0 = test.data[0]
    plt.imshow(pic_0)
    plt.show()
