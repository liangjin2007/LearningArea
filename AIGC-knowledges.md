## 基础
深度学习框架学习 https://github.com/dragen1860/Deep-Learning-with-PyTorch-Tutorials.git

### Lesson01 PyTorch初见
```

框架
  Google: tensorflow, keras, theano
  Amazon: mxnet, gluon
  Microsoft: cntk, chainer
  Facebook: caffe, caffe2, pytorch, torch7,


Pytorch学术界用得比较多
Tensorflow工业界用得比较多

Pytorch生态： Pytorch NLP, Pytorch geometry, Torch Vision, Fast.ai, ONNX

Pytorch能做什么： GPU加速， 自动求导， 常用网络层
nn.Linear, nn.Conv2d, nn.LSTM, nn.ReLU, nn.Sigmoid, nn.Softmax, nn.CrossEntropyLoss, nn.MSE

# 自动求导autograd_demo.py
import  torch
from    torch import autograd
x = torch.tensor(1.)
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = torch.tensor(3., requires_grad=True)
y = a**2 * x + b * x + c
print('before:', a.grad, b.grad, c.grad)
grads = autograd.grad(y, [a, b, c])
print('after :', grads[0], grads[1], grads[2])

# GPU加速 gpu_accelerate.py
import 	torch
import  time
print(torch.__version__)
print(torch.cuda.is_available())
# print('hello, world.')
a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)
t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print(a.device, t1 - t0, c.norm(2))
device = torch.device('cuda')
a = a.to(device)
b = b.to(device)
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))
```

### Lesson02 安装
Anaconda3, python version, cuda version, python develop IDE(VSCode/Pycharm)

### Lesson03 简单回归案例
```
线性回归 y = wx + b + eps
loss = sum_i((wxi + b - yi)^2)
```

### Lesson04 简单回归案例实战
```
import numpy as np

# y = wx + b
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = w_current - (learningRate * w_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]

def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
          .format(initial_b, initial_m,
                  compute_error_for_line_given_points(initial_b, initial_m, points))
          )
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, m,
                 compute_error_for_line_given_points(b, m, points))
          )

if __name__ == '__main__':
    run()
```

### MNIST
- utils.py
```
import  torch
from    matplotlib import pyplot as plt
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()
def plot_image(img, label, name):

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()
def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out
```

- 数据集loader及遍历器
```
import  torch
import torchvision
from    utils import plot_image, plot_curve, one_hot
batch_size = 512
# step1. load dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')
```

- Network 定义
```
from    torch import nn  # 网络框架在这里
from    torch.nn import functional as F  # 激活函数在这里, loss也在这个库里
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()

        # xw+b
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = self.fc3(x)

        return x

问题：
反向不用写么？ 由autograd搞定
```

- 定义网络、优化器，写迭代过程训练， 绘制训练Loss曲线， 测试
```
net = Net()
# [w1, b1, w2, b2, w3, b3], net.parameters()返回这个? 
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []
for epoch in range(3):

    for batch_idx, (x, y) in enumerate(train_loader):

        # x: [b, 1, 28, 28], y: [512]
        # [b, 1, 28, 28] => [b, 784]
        x = x.view(x.size(0), 28*28)   # Tensor.view ? 
        # => [b, 10]
        out = net(x)
        # [b, 10]
        y_onehot = one_hot(y)          # 右侧为整数，左侧为[0,..., 1, 0, ..., 0]
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot) # 定义mse loss

        optimizer.zero_grad()   # 初始化梯度w.grad为0
        loss.backward()         # backward传递， 得到每个w.grad
        # w' = w - lr*grad
        optimizer.step()        # 更新w = w - lr * grad或者其他公式，这取决于optimizer。

        train_loss.append(loss.item()) # 添加loss

        if batch_idx % 10==0:
            print(epoch, batch_idx, loss.item())

# 绘制训练曲线
plot_curve(train_loss)


# 测试，得到测试准确率
total_correct = 0
for x,y in test_loader:
    x  = x.view(x.size(0), 28*28)
    out = net(x)
    # out: [b, 10] => pred: [b]
    pred = out.argmax(dim=1)        # 得到最大值的index
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

# 稍微有点不明白
x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28*28)) # 前向预测？
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')

```

### 基本数据类型
```

```


### 网络架构
### 网络层
### Loss

### 优化器
### 训练
### 预测
### 部署
### 网络减枝

## [awesome-visual-transformer]( https://github.com/dk-liang/Awesome-Visual-Transformer )
## [awesome-avatars](https://github.com/pansanity666/Awesome-Avatars?tab=readme-ov-file)
## [awesome-3d-AIGC](https://github.com/mdyao/Awesome-3D-AIGC)
## [awesome-autonomous-vehicle](https://github.com/DeepTecher/awesome-autonomous-vehicle)
## [awesome-3d-generation](https://github.com/justimyhxu/awesome-3D-generation)
## [awesome-3d-diffusion](https://github.com/cwchenwang/awesome-3d-diffusion)
## [awesome-3d-guassiansplatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)
### Seminar paper [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting)
- 安装
```
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
conda config --add pkgs_dirs F:/pkgs
conda env create --file ./envronment.yml --prefix F:/envs/gaussian_splatting
conda activate F:/envs/gaussian_splatting
```

-- 看论文/读代码
