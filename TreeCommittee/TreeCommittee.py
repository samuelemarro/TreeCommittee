import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import better_resnet

#TODO: Lo shuffle della validazione non è necessaria
#TODO: Copiare anche le varie migliorie/ottimizzazioni di ResNet
#TODO: Testare con dei partial_output più raffinati

def get_loaders(num_workers, pin_memory, validation_ratio=0):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
    
    indices = list(range(len(trainset)))
    split = int(np.floor(len(trainset) * validation_ratio))
    train_indices, validation_indices = indices[split:], indices[:split]

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=40,
                                          shuffle=False, num_workers=num_workers,
                                          sampler=torch.utils.data.sampler.RandomSampler(train_indices),
                                          pin_memory=pin_memory)

    validation_loader = torch.utils.data.DataLoader(trainset, batch_size=20,
                                         shuffle=False, num_workers=num_workers,
                                         sampler=torch.utils.data.sampler.RandomSampler(validation_indices),
                                         pin_memory=pin_memory)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=20,
                                         shuffle=False, num_workers=num_workers,
                                         pin_memory=pin_memory)

    return train_loader, validation_loader, test_loader


def train_multiple(network_sets, optimizer_sets, criterion, train_loader, train_epochs, validation_loader, test_loader):
    total_time = 0
    results = []
    best_validation_accuracy = -1
    best_test_accuracy = -1

    for network_set, optimizer_set in zip(network_sets, optimizer_sets):
        for epoch in range(train_epochs):
            for network in network_set:
                network.train()
                network.cuda()

            start_time = time.perf_counter()

            for data in train_loader:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))

                for network, optimizer in zip(network_set, optimizer_set):
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = network.forward(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()

                    optimizer.step()

            elapsed_time = time.perf_counter() - start_time
            total_time += elapsed_time

            for network in network_set:
                network.eval()
                network.cpu()

            for network in network_set:
                #Get the validation_accuracy
                validation_accuracy = get_accuracy(network, validation_loader)

                #If the validation accuracy is a new record, save it along with the test accuracy
                #Note: A high validation accuracy is not always correlated with a high test accuracy

                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_test_accuracy = get_accuracy(network, test_loader)

            print()
            print('Time: {}'.format(total_time))
            print('Best Validation Accuracy: {}%'.format(best_validation_accuracy * 100.0))
            print('Corresponding Test Accuracy: {}%'.format(best_test_accuracy * 100.0))

            results.append((total_time, best_test_accuracy))
    
    return results



class ResizeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__(in_features, out_features, bias)

    def forward(self, input):
        return super().forward(input.view(input.shape[0], self.in_features))

class PooledConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, conv_stride=1, conv_padding=0, pool_stride=None, pool_padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, conv_stride, conv_padding)
        self.pool = nn.MaxPool2d(pool_size, pool_stride, pool_padding)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class Detacher(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
    def forward(self, x):
        return self.network.forward(x.detach())

def base(module_functions, network_function, n, criterion, train_loader, train_epochs):
    nets = []

    for i in range(n):
        net = network_function([x() for x in module_functions])
        nets.append(net)

    optimizers = [torch.optim.Adam(x.parameters()) for x in nets]

    return nets, [nets], [optimizers]

def build_tree(modules, current_module_functions, branches):
        collections = []
        current_module = current_module_functions[0]()

        collections.append(modules + [current_module])

        if len(current_module_functions) > 1:
            for i in range(branches):
                collections += build_tree(modules + [current_module], current_module_functions[1:], branches)

        return collections

def tree(module_functions, network_function, partial_output_functions, branches):
    split_training_networks, split_training_parameter_sets, final_networks = make_tree(module_functions, network_function, partial_output_functions, branches)

    optimizer_sets = []
   
    for parameter_sets in split_training_parameter_sets:
        optimizer_sets.append([torch.optim.Adam(parameter_set) for parameter_set in parameter_sets])

    return final_networks, split_training_networks, optimizer_sets

def direct_tree(module_functions, network_function, branches, criterion, train_loader, train_epochs):
    collections = build_tree([], module_functions, branches)

    max_depth = max([len(x) for x in collections])

    final_networks = [network_function(x) for x in collections if len(x) == max_depth]

    optimizers = [torch.optim.Adam(x.parameters()) for x in final_networks]

    return final_networks, final_networks, optimizers

def forward_thinking(module_functions, network_function, partial_output_functions, n):
    training_network_sets = [[] for i in range(len(module_functions))]
    optimizer_sets = [[] for i in range(len(module_functions))]

    for i in range(n):
        networks, parameter_sets = make_forward_thinking(module_functions, network_function, partial_output_functions)
        optimizers = [torch.optim.Adam(x) for x in parameter_sets]

        for j, (network, optimizer) in enumerate(zip(networks, optimizers)):
            training_network_sets[j].append(network)
            optimizer_sets[j].append(optimizer)

    return training_network_sets[-1], training_network_sets, optimizer_sets

def make_forward_thinking(module_functions, network_function, partial_output_functions):
    split_networks = []
    previous_collection = []

    parameter_sets = []

    for i, module_function in enumerate(module_functions):
        module = module_function()

        parameter_set = []
        parameter_set += module.parameters()
        
        if i == len(module_functions) - 1:
            current_collection = previous_collection + [module]
            split_networks.append(network_function(previous_collection + [module]))
        else:
            partial_output_module = partial_output_functions[i]()
            current_collection = previous_collection + [module] + [partial_output_module]
            parameter_set += partial_output_module.parameters()
            split_networks.append(nn.Sequential(network_function(previous_collection + [Detacher(module)]), partial_output_module))

        parameter_sets.append(parameter_set)
        previous_collection.append(module)
    
    return split_networks, parameter_sets

def make_tree(module_functions, network_function, partial_output_functions, branches):
    tree_collections = build_tree([], module_functions, branches)

    max_depth = max([len(x) for x in tree_collections])

    final_networks = [network_function(x) for x in tree_collections if len(x) == max_depth]

    sorted_training_networks = [[] for i in range(max_depth)]
    sorted_training_parameter_sets = [[] for i in range(max_depth)]

    for collection in tree_collections:
        parameter_set = []

        parameter_set += collection[-1].parameters()

        effective_length = len(collection)

        if effective_length < max_depth:
            output_module = partial_output_functions[len(collection) - 1]()

            parameter_set += output_module.parameters()
            network = nn.Sequential(network_function(collection[:-1] + [Detacher(collection[-1])]), output_module)
        else:
            network = network_function(collection[:-1] + [Detacher(collection[-1])])

        sorted_training_networks[effective_length - 1].append(network)
        sorted_training_parameter_sets[effective_length - 1].append(parameter_set)

    return sorted_training_networks, sorted_training_parameter_sets, final_networks

def get_accuracy(network, test_loader):
    total = 0
    correct = 0
    
    network.cuda()

    for data in test_loader:
        images, labels = data
        images = images.cuda(async=True)
        labels = labels.cuda(async=True)
        outputs = network(Variable(images))

        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)

        correct += (predicted == labels).sum()
    
    network.cpu()

    return correct / total

def get_custom_resnet(block, num_blocks, num_classes=10):
    def make_layer(block, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, planes, stride))
            in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    in_planes = [64, 64 * block.expansion, 128 * block.expansion, 256 * block.expansion, 512 * block.expansion]

    module_functions = [
        lambda: nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU()),
        lambda: make_layer(block, in_planes[0], 64, num_blocks[0], stride=1),
        lambda: make_layer(block, in_planes[1], 128, num_blocks[1], stride=2),
        lambda: make_layer(block, in_planes[2], 256, num_blocks[2], stride=2),
        lambda: nn.Sequential(make_layer(block, in_planes[3], 512, num_blocks[3], stride=2),
                              nn.AvgPool2d(4),
                              ResizeLinear(in_planes[4], num_classes))
        ]

    partial_output_functions = [
        lambda: ResizeLinear(in_planes[0] * 32 * 32, num_classes),
        lambda: ResizeLinear(in_planes[1] * 32 * 32, num_classes),
        lambda: ResizeLinear(in_planes[2] * 16 * 16, num_classes),
        lambda: ResizeLinear(in_planes[3] * 8 * 8, num_classes)
        ]

    network_function = lambda modules: nn.Sequential(*modules)

    return module_functions, partial_output_functions, network_function

def get_resnet(resnet_number, num_classes=10):
    if resnet_number == 18:
        return get_custom_resnet(better_resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif resnet_number == 34:
        return get_custom_resnet(better_resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    elif resnet_number == 50:
        return get_custom_resnet(better_resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    elif resnet_number == 101:
        return get_custom_resnet(better_resnet.Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    elif resnet_number == 152:
        return get_custom_resnet(better_resnet.Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    else:
        raise ValueError('Invalid ResNet number. Valid: [18, 34, 50, 101, 152]')

def get_convolutional():
    module_functions = [
        lambda: PooledConvolution(3, 64, 2, 2, conv_stride=2, pool_stride=2),
        lambda: PooledConvolution(64, 128, 2, 2, conv_stride=2, pool_stride=2),
        lambda: ResizeLinear(128 * 2 * 2, 200),
        lambda: nn.Linear(200, 200),
        lambda: nn.Linear(200, 10)
        ]
    partial_output_functions = [
        lambda: ResizeLinear(64 * 8 * 8, 10),
        lambda: ResizeLinear(128 * 2 * 2, 10),
        lambda: nn.Linear(200, 10),
        lambda: nn.Linear(200, 10)
        ]
    
    network_function = lambda modules: nn.Sequential(*modules)

    return module_functions, partial_output_functions, network_function

def split_for_memory(training_network_sets, training_optimizer_sets):
    networks = []
    optimizers = []
    for network_set, optimizer_set in zip(training_network_sets, training_optimizer_sets):
        networks += network_set
        optimizers += optimizer_set

    return [[x] for x in networks],  [[x] for x in optimizers]

def main(flags):
    module_functions, partial_output_functions, network_function = get_resnet(18)

    #2, False or 1, True
    train_loader, validation_loader, test_loader = get_loaders(0, True, 0.1)
    criterion = nn.CrossEntropyLoss()

    branches = 2

    total_networks = branches**(len(module_functions) - 1)

    train_epochs = [10]#[20]#[1, 2, 5, 10, 15, 20]

    tree_accuracies = []
    tree_times = []

    base_accuracies = []
    base_times = []

    #TODO: Testare single

    for train_epoch_number in train_epochs:
        #Tree: Una rete addestrata progressivamente con ramificazioni
        final_networks, training_network_sets, training_optimizer_sets = tree(module_functions, network_function, partial_output_functions, branches)
        print('Training Tree')
        
        #Riorganizza per ridurre la memoria sacrificando la velocità
        training_network_sets, training_optimizer_sets = split_for_memory(training_network_sets, training_optimizer_sets)
        results = train_multiple(training_network_sets, training_optimizer_sets, criterion, train_loader, train_epoch_number, validation_loader, test_loader)

        times = [x[0] for x in results]
        accuracies = [x[1] * 100.0 for x in results]

        tree_accuracies += accuracies
        tree_times += times

        #Base: Tante reti poco addestrate
        final_networks, training_network_sets, training_optimizer_sets = base(module_functions, network_function, total_networks, criterion, train_loader, train_epoch_number)
        print('Training Base')
        
        #Riorganizza per ridurre la memoria sacrificando la velocità
        training_network_sets, training_optimizer_sets = split_for_memory(training_network_sets, training_optimizer_sets)
        results = train_multiple(training_network_sets, training_optimizer_sets, criterion, train_loader, train_epoch_number, validation_loader, test_loader)

        times = [x[0] for x in results]
        accuracies = [x[1] * 100.0 for x in results]

        base_accuracies += accuracies
        base_times += times

    plt.plot(tree_times, tree_accuracies)
    plt.plot(base_times, base_accuracies)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', nargs='?', const=True, type=bool, default=True, help='If True, uses a CUDA-enabled GPU (if available)')
    flags, unparsed = parser.parse_known_args()

    main(flags)
