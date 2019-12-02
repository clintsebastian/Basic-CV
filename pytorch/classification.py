import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        # Conv2d: input channels=1, output_channels=32, kernel=(3, 3), stride=1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU(inplace=True)  # Check what inplace does
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*5*5, 256)
        self.fc2 = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: input 28x28x1
        x = self.conv1(x)       # conv1: 26x26x32
        x = self.relu1(x)       # relu1: 26x26x32
        x = self.pool1(x)       # pool1: 11x11x32
        x = self.conv2(x)       # conv2: 11x11x64
        x = self.relu2(x)       # relu1: 11x11x64
        x = self.pool2(x)       # pool2: 5x5x64
        x = x.view(-1, 5*5*64)  # reshape into a 1D array (1600)
        x = self.fc1(x)         # fc1: 1600x256 -> 256
        x = self.fc2(x)         # fc2: 256x10 -> 10
        return F.log_softmax(x, dim=1)


def mnist_data_loader(path, train=True, batch_size=10):
    """
    :param path: Path to save MNIST dataset
    :param train: boolean, load data to train or test.
    :return: data loader
    """
    # Inputs are converted to pytorch tensor, normalized.
    transformation = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,),)])

    # Inbuilt dataset class
    mnist_dataset = datasets.MNIST(root=path,
                                   download=True,
                                   train=train,
                                   transform=transformation)

    data_loader = data.DataLoader(mnist_dataset, batch_size=batch_size)
    return data_loader


def train_model(data_loader, model, epoch=10):

    # To train the model, set model to train.
    model.train()
    # Use Stochastic Gradient Descent Optimizer
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i in range(epoch):
        for idx, (input_data, label) in enumerate(data_loader):
            opt.zero_grad()  # Set gradient to zero after each iteration
            pred = model(input_data)
            # Negative log likelihood loss
            loss = F.nll_loss(pred, label)
            loss.backward()  # Back propagation
            opt.step()
            if idx % 100 == 0:
                print('Iteration: {}, Loss: {}'.format(idx, round(loss.item(), 2)))

        print('Epoch completed {}'.format(i))
        if i % 1 == 0:
            if not os.path.exists('mnist_model/'):
                os.makedirs('mnist_model/')
            torch.save(model.state_dict(), 'mnist_model/epoch_{}.pth'.format(i))
            print('Model saved at epoch {}'.format(i))
            test_loader = mnist_data_loader(data_path, False, batch_size=30)
            test_model(test_loader, model)


def test_model(data_loader, model):
    """
    Test the model
    :param data_loader:
    :param model:
    :return: nothing, prints accuracy.
    """
    # To evaluate the model, we set model.eval()
    model.eval()
    correct = 0
    for idx, (input_data, label) in enumerate(data_loader):
        output = model(input_data)
        # Get the position with the highest probability
        pred = output.argmax(dim=1)
        # Check if prediction equals to label
        correct += pred.eq(label.view_as(pred)).sum().item()

    total_samples = len(data_loader.dataset)
    print('Accuracy: {}'.format(correct * 100.0 / total_samples))


def denormalize(input_image, mean=0.1307, std=0.3081):
    """
    The image is fed for training after setting to zero mean and unit variance.
    For visualization, we need to set image values between 0 and 255.
    :param input_image: Input image that has 0 mean, unit variance
    :param mean:
    :param std:
    :return: image as unsigned 8 bit with non-active dimensions removed
    """
    image = (input_image * std + mean) * 255
    return image.astype(np.uint8).squeeze()


def get_random_sample_in_batch(data_loader, batch_size):
    random_idx = random.randint(0, batch_size)
    input_data, label = [x[random_idx] for x in next(iter(data_loader))]
    return input_data.unsqueeze(dim=0), label


def test_random_image(data_loader, model, epoch, batch_size):
    model.eval()
    state = torch.load('mnist_model/epoch_{}.pth'.format(epoch))
    model.load_state_dict(state)

    if not os.path.exists('mnist_model/results/'):
        os.makedirs('mnist_model/results/')

    input_data, label = get_random_sample_in_batch(data_loader, batch_size)
    output = model(input_data)
    pred = output.argmax(dim=1)
    image = input_data.cpu().numpy()
    image = denormalize(image)

    file_name = 'random_image.png'
    cv2.imwrite('mnist_model/results/' + file_name, image)
    root_path = os.path.dirname(os.path.abspath(__file__))
    result_dir = root_path + '/mnist_model/results/'
    print('Check for the random image in: {}'.format(result_dir))
    print('Label: {}, Prediction: {}'.format(label, pred.item()))


if __name__ == '__main__':
    data_path = '../mnist'
    mnist_loader = mnist_data_loader(data_path, True, batch_size=64)
    simple_net = SimpleNet()
    # train_model(mnist_loader, simple_net)
    test_random_image(mnist_loader, simple_net, 9, 100)


