import torch
import torch.nn as nn
from datasets.load_datasets import load_CIFAR
from models.conv_models import ResNet18
from models.linear_models import LinearModel
import torchvision.transforms as transforms
from utils.exp import getDataLoader, train, test
from data_loader import Caltech101
import matplotlib.pyplot as plt


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    """
        dataset and loader
    """

    # trainset, testset = load_CIFAR(root_path="../datasets/CIFAR", transform=transform, download=False)
    # trainset, testset = load_MNIST(root_path="../datasets", transform=transform, download=True)
    # train_loader = getDataLoader(trainset)
    # test_loader = getDataLoader(testset, train=False)

    train_set, val_set, test_set = Caltech101(flag='train'), Caltech101(flag='val'), Caltech101(flag='test')
    train_loader, val_loader, test_loader = getDataLoader(train_set, flag='train'), getDataLoader(val_set, flag='val'), getDataLoader(train_set, flag='test')


    # model = ConvNet(in_channel=3).to(device)
    # model = ResNet18().to(device)
    model = LinearModel(num_hidden=120000, classes=102).to(device)

    criterion = nn.NLLLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    print('-----------------train start-----------------')
    train(train_loader, val_loader, test_loader, model, epochs=15, criterion=criterion, optimizer=optimizer, device=device)

    print('-----------------test start-----------------')
    test(test_loader, model, criterion, device)

    # weight visualization
    weight = model.linear.weight.detach().cpu().numpy()
    print(weight.shape)
    weight = weight.reshape(-1, 200, 200, 3)
    N = weight.shape[0]
    rows = 2
    for i in range(N):
        if i != 1:
            continue
        max_w = weight[i].max(0).max(0)[None, None, :]
        min_w = weight[i].min(0).min(0)[None, None, :]
        # print(max_w.shape)
        tmp = (weight[i] - min_w) / (max_w - min_w)
        # plt.subplot(N // rows, rows, i + 1)
        plt.imshow(tmp)
        break
    plt.show()



