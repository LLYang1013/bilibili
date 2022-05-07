import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #  前面的（0.5，0.5，0.5） 是 R G B 三个通道上的均值， 后面(0.5, 0.5, 0.5)是三个通道的标准差

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #  显示4张图片看一下，显示的时候，将val_loader中的bs变成4
    '''def imshow(img):
        img = img / 2 + 0.5  # 反标准化：除方差0.5，+均值0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # tensor(c,h,w)--->numpy(h,w,c)
        plt.show()

    # print labels
    print(' '.join(f'{classes[val_label[j]]:5s}' for j in range(4)))
    # show images
    imshow(torchvision.utils.make_grid(val_image))'''

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):  # loop over the dataset multiple times扫5次

        running_loss = 0.0  # 用来累加损失
        for step, data in enumerate(train_loader, start=0):  # 使用enumerate函数，返回每一批的data，还返回相应的索引step
            #  data 是列表 [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()  # 每计算一个batch，就要清空一次梯度
            # forward + backward + optimize
            outputs = net(inputs)  # 输入图片
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:  # print every 500 mini-batches
                with torch.no_grad():  # 上下文管理器，下面的不计算梯度
                    outputs = net(val_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]
                    # 在dim=1第一个维度找最大，也就是10类中。[1] 表示index
                    # torch.max(a, 1): 返回每一行的最大值，且返回索引
                    # torch.max()[0]: 只返回最大值
                    # torch.max()[1]: 只返回最大值的索引

                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                    # predict_y == val_label 返回True或False，1或0；sum()是求共有多少个预测正确的
                    # item() 是将前面的tensor转换成数值

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    # 第epoch轮的第step步， 500步中平均的训练误差，准确率
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
