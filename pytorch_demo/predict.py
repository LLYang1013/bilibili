import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))  # 训练出来的LeNet的权重参数

    # 图片有点问题：读取的是.png图片是32位深度的，一般我们读取的是.jpg 24位深度的，转换为RGB就行了
    im = Image.open('1.jpg').convert('RGB')
    # im = Image.open('1.jpg')  # 通过pil或numpy导入的图片[h,w,c]
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [bs, C, H, W] #增加一个维度bs

    with torch.no_grad():
        outputs = net(im)
        # predict = torch.max(outputs, dim=1)[1].numpy()
    # print(classes[int(predict)])   # 这种方法输出  类别dog
        predict = torch.softmax(outputs, dim=1)
    print(predict)  # 这种方法输出每种类别预测的概率
    # tensor([[1.2230e-02, 9.8317e-05, 2.3522e-02, 2.1442e-01, 7.1492e-03, 6.8609e-01,
    #          1.1974e-02, 4.1822e-02, 6.0769e-04, 2.0857e-03]]) 可见狗的概率更大6.8609e-01，猫其次2.1442e-01


if __name__ == '__main__':
    main()
    #  虽然有猫和狗，它输出一个 dog，可能占比更大吧
