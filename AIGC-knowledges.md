![image](https://github.com/liangjin2007/LearningArea/assets/4059558/1ffae0ce-e220-4277-bb40-e80547b72f96)## 基础
Deep-Learning-with-PyTorch-Tutorials https://github.com/dragen1860/Deep-Learning-with-PyTorch-Tutorials.git

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

### Lesson05 MNIST
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

### Lesson06 基本数据类型
![data type](https://github.com/liangjin2007/data_liangjin/blob/master/pytorch_data_type.png?raw=true)
- Types
```
torch.float32, torch.float,
torch.float64, torch.double,
torch.int, torch.int32,
torch.int8,
torch.uint8,
torch.float16, torch.half,
torch.int16, torch.short,
torch.int64, torch.long
```


- cpu gpu tensors
```
torch.FloatTensor, torch.cuda.FloatTensor
torch.DoubleTensor, torch.cuda.DoubleTensor
torch.IntTensor, torch.cuda.IntTensor
torch.CharTensor, torch.cuda.CharTensor
torch.ByteTensor, torch.cuda.ByteTensor
torch.HalfTensor, torch.cuda.HalfTensor
torch.ShortTensor, torch.cuda.ShortTensor
torch.LongTensor, torch.cuda.LongTensor
```

- Type Check
```
a = torch.randn(2, 3)
a.type() # 'torch.FloatTensor'
type(a) # 'torch.Tensor'
isinstance(a, torch.FloatTensor) # True

data = data.cuda()
isinstance(data, torch.cuda.DoubleTensor) # true
```

- Dimension / rank
```
// Dim 0 / range 0
a = torch.tensor(1.0)
a.shape         # output torch.Size([])
len(a.shape)    # output 0
a.size()        # torch.Size([])


// Dim 1 / rank 1
torch.tensor([1.0])
torch.tensor([1.1, 2.2])   # tensor([1.1, 2.2])

// 这个有点类似np.ones(2)
torch.FloatTensor(1)       # parameter must be a int or a sequence.   output tensor([0.0])
torch.FloatTensor(2)       # parameter must be a int or a sequence.   output tensor([0.0, 3.0])
data = np.ones(2)          # output array([1., 1.])
torch.from_numpy(data)     # convert numpy array to tensor, tensor([1., 1.], dtype=torch.float64)


// Dim 1
a = torch.ones(2)         #
a.shape                   # output torch.Size([2])

// Dim 2
a = torch.randn(2, 3)
a.shape                   # output torch.Size([2, 3])
a.size(0)                 # output 2
a.size(1)                 # output 3
a.shape[1]                # output 3

// Dim 3
a = torch.rand(1, 2, 3)
a.shape                   # output torch.Size([1, 2 ,3])
list(a.shape)             # output [1, 2, 3]

// Dim 4
a = torch.rand(2, 3, 28, 28)
a.shape                   # output torch.Size([2, 3, 28, 28])
a.numel()                 # return 2 * 3 * 28 * 28
a.dim()                   # return 4
a = torch.tensor(1)
a.dim()                   # 0
```    

### Lesson07 创建Tensor
```
// import from numpy
a = np.array([2, 3.3])
torch.from_numpy(a)      # tensor([2.000, 3.000], dtype = torch.float64)

// import from list
a = torch.tensor([2.0, 3.3])
b = torch.FloatTensor([2.0, 3.3])   # 注意，也由一种是直接传整数参数，这种是创建Dim 1的tensor, 长度为传入的整数
d = torch.FloatTensor(2, 3)         # 创建 2 x 3 张量
c = torch.tensor([[2.0, 3.0], [4.0, 5.0]]) # 二维

torch.tensor是传内容来创建
torch.Tensor(2, 3)                  # 注意看torch.tensor和torch.Tensor的区别。 torch.Tensor应该是和torch.FloatTensor一样的用法，可以传内容，也可以指定维度。



// 创建uninitialized张量
torch.empty(1)                      # [0.0]
torch.Tensor(2, 3)                  # 
torch.FloatTensor(2, 3)


// Set torch.tensor的默认类型
torch.set_default_tensor_type(torch.DoubleTensor)

//rand/rand_like, randint
a = torch.rand(3, 3)
b = torch.rand_like(a)
torch.randint

// randn N(0, 1)
torch.randn(3, 3)

// N(u, std)
torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1))

// full
torch.full([2, 3], 7.0)  # [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]]
torch.full([], 7.0)      # tensor(7.0)
torch.full([1], 7.0)     # tensor([7.0])

// arange/range
torch.arange(0, 10)      # tensor([0, 1, 2, ..., 9])
torch.arange(0, 10, 2)   # tensor([0, 2, 4, 6, 8])
torch.range(0, 10) 

// linspace/logspace
torch.linspace(0, 10, steps = 4)    # tensor([0.0000, 3.3333, 6.6666, 10.0000])
torch.linspace(0, 10, steps = 10)   # tensor([0.0000, 1.1111, 2.2222, ..., 10.0])

// ones/zeros/eye/ones_like

// randperm   ... random.shuffle
```

### Lesson08 索引与切片 Index / Slicing
```
// Index
a = torch.rand(4, 3, 28, 28)
a[0].shape                     # torch.Size([3, 28, 28])
a[0, 1].shape                  # torch.Size([28, 28])
a[0, 0, 2, 4]                  # tensor(0.8042)

// First/last n
a[:2].shape                          # torch.Size([2, 3, 28, 28])
a[:2, :1, :, :]                      # :2, 不包括2, torch.Size([2, 1, 28, 28])
a[:2, 1:, :, :]                      # 1:, 包括1， torch.Size([2, 2, 28, 28])
a[:2, -1:, :, :]                     # -1:, the last index, torch.Size([2, 1, 28, 28])

// Select by steps
a[:, :, 0:28:2, 0:28:2]              # [4, 3, 14, 14]
a[:, :, ::2, ::2]                    # [4, 3, 14, 14]

// Select by specific index
a.index_select(2, torch.arange(8)).shape    # [4, 3, 8, 28]

// ...
a[...].shape            # [4, 3, 28, 28]
a[0, ...].shape         # [3, 28, 28]
a[:, 1, ...].shape      # [4, 28, 28]
a[..., :2].shape        # [4, 3, 28, 2]

// Select by mask
x = torch.randn(3, 4)
mask = x.ge(0.5)
torch.masked_select(x, mask)  # tensor([0.5404, 0.6040, 1.5771])
torch.masked_select(x, mask).shape # torch.Size([3])

// Select by flatten index
src = torch.tensor([[4, 3, 5], [6, 7, 8]])
torch.take(src, torch.tensor([0, 2, 5]))   # tensor([4, 5, 8])
```

### Lesson09 维度变换
```
// View/reshape
a = torch.rand(4, 1, 28, 28)
a.view(4, 28*28)   # [4, 784]
a.view(4*28, 28)   # [112, 28]
a.view(4*1, 28, 28) # [4, 28, 28]
b = a.view(4, 784)
b.view(4, 28, 28, 1) # logic bug ????

// Squeeze/unsqueeze
a.unsqueeze(0).shape         # [1, 4, 1, 28, 28]
a.unsqueeze(-1).shape        # [4, 1, 28, 28, 1]
a.unsqueeze(-4).shape        # [4, 1, 1, 28, 28]
a.unsqueeze(-5).shape        # [1, 4, 1, 28, 28]
a.unsqueeze(5).shape         # RuntimeError
Valid parameter range : [-a.dim()-1, a.dim()+1)

b = torch.rand(32)
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)      # [1, 32, 1, 1]
b.squeeze().shape                                 # [32]
b.squeeze(0).shape                                # [32, 1, 1]
b.squeeze(-1).shape                               # [1, 32, 1]
b.squeeze(1).shape                                # [1, 32, 1, 1]
b.squeeze(-4).shape                               # [32, 1, 1]


// Transpose/t/permute
a = torch.Tensor(4, 3, 32, 32)
a1 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 3, 32, 32)
a2 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 32, 32, 3).transpose(1, 3)
torch.all(torch.eq(a, a1))    # tensor(0, dtype = torch.uint8)
torch.all(torch.eq(a, a2))    # tensor(1, dtype = torch.uint8)

b = torch.rand(4, 3, 28, 32)
b.permute(0, 2, 3, 1)              # [4, 28, 32, 3]

// Expand
b.expand ???

// Repeat
b = torch.Tensor(1, 32, 1, 1)
b.repeat(4, 1, 1, 1)            # [4, 32, 1, 1]
```

### Lesson10 Broadcasting
![broadcasting](https://github.com/liangjin2007/data_liangjin/blob/master/broadcasting.png?raw=true)

### Lesson11 拼接与拆分
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
- github代码没安装成功，转而切到c++版的一个实现https://github.com/MrNeRF/gaussian-splatting-cuda
#### gaussian-splatting-cuda
```
官方只配过linux系统， windows系统需要趟个坑。
```

- 安装开发环境：
```
安装Visual Studio 2019 Community
安装NSight Compute-win64-2021.2.1.xxx
安装CUDA 11.8
安装NVIDIA_NSight_Visual_Studio_Edition
安装CMake
安装Python3.9
如果Visual Studio找不到CUDA（比如打不开依赖于CUDA的project）, 拷贝 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vxxx\extras\visual_studio_integration\MSBuildExtensions中的四个文件 到 C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations
```

- 配置工程
```
// Step 1.
在目标位置右键打开git Bash Here，敲下面命令
git clone --recursive https://github.com/MrNeRF/gaussian-splatting-cuda

// Step 2.
从pytorch官网去下载windows版libtorch 2.0.1， 解压放到external/libtorch
下载tbb库放到external/tbb
拷贝Python安装目录中的include, libs, 和python39.dll拷贝到external/pythonlib3.9

// Step 3.
修改CMakeLists.txt：
  添加变量
    set(PYTHONLIB_INCLUDE_DIR "${PROJ_ROOT_DIR}/external/pythonlib3.9/include")
    set(PYTHONLIB_LIBRARY "${PROJ_ROOT_DIR}/external/pythonlib3.9/libs/python39.lib")
  target_include_directories(${PROJECT_NAME} xxx ${PYTHONLIB_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME}
        PRIVATE
        xxx 
        ${PYTHONLIB_LIBRARY}

// Step 4. 修改一处编译不过的地方
    // Set up rasterization configuration
    GaussianRasterizationSettings raster_settings = {
        //.image_height = static_cast<int>(viewpoint_camera.Get_image_height()),
        //.image_width = static_cast<int>(viewpoint_camera.Get_image_width()),
        //.tanfovx = std::tan(viewpoint_camera.Get_FoVx() * 0.5f),
        //.tanfovy = std::tan(viewpoint_camera.Get_FoVy() * 0.5f),
        //.bg = bg_color,
        //.scale_modifier = scaling_modifier,
        //.viewmatrix = viewpoint_camera.Get_world_view_transform(),
        //.projmatrix = viewpoint_camera.Get_full_proj_transform(),
        //.sh_degree = gaussianModel.Get_active_sh_degree(),
        //.camera_center = viewpoint_camera.Get_camera_center(),
        //.prefiltered = false
        static_cast<int>(viewpoint_camera.Get_image_height()),
        static_cast<int>(viewpoint_camera.Get_image_width()),
        std::tan(viewpoint_camera.Get_FoVx() * 0.5f),
        std::tan(viewpoint_camera.Get_FoVy() * 0.5f),
        bg_color,
        scaling_modifier,
        viewpoint_camera.Get_world_view_transform(),
        viewpoint_camera.Get_full_proj_transform(),
        gaussianModel.Get_active_sh_degree(),
        viewpoint_camera.Get_camera_center(),
        false
    };

// Step 5. CMake GUI设置 Where is the source code 和 Where to build the binaries, 点Configure, 会报错，提示tbb找不到。设置好tbb，Configure成功。点Generate成功。打开Visual Studio 2019
```

- 下载数据
```  
从 https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip 下载tandt_db.zip, unzip到data目录中。
得到:
  data/db
  data/tandt
```

- 命令行执行
```
Visual Studio启动方式：
  在Visual Studio 的项目属性->调试->命令参数 中设置 -d "xxx/gaussian-splatting-cuda/data/tandt/train" -o "xxx/gaussian-splatting-cuda/output/tandt/train"
  在Visual Studio 的项目属性->调试->环境 中添加 PATH=%PATH%;path/to/cuda/bin;path/to/pythonlib3.9; path/to/libtorch/lib

```
- 所有命令行选项
```
命令行选项
Core Options
-h, --help
Display this help menu.

-d, --data_path [PATH]
Specify the path to the training data.

-f, --force
Force overwriting of output folder. If not set, the program will exit if the output folder already exists.

-o, --output_path [PATH]
Specify the path to save the trained model. If this option is not specified, the trained model will be saved to the "output" folder located in the root directory of the project.

-i, --iter [NUM]
Specify the number of iterations to train the model. Although the paper sets the maximum number of iterations at 30k, you'll likely need far fewer. Starting with 6k or 7k iterations should yield preliminary results. Outputs are saved every 7k iterations and also at the end of the training. Therefore, even if you set it to 5k iterations, an output will be generated upon completion.

Advanced Options
--empty-gpu-cache Empty CUDA memory after ever 100 iterations. Attention! This has a considerable performance impact

--enable-cr-monitoring
Enable monitoring of the average convergence rate throughout training. If done, it will stop optimizing when the average convergence rate is below 0.008 per default after 15k iterations. This is useful for speeding up the training process when the gain starts to dimish. If not enabled, the training will stop after the specified number of iterations --iter. Otherwise its stops when max 30k iterations are reached.

-c, --convergence_rate [RATE]
Set custom average onvergence rate for the training process. Requires the flag --enable-cr-monitoring to be set.
```

- 修复crash
```
cuda debug

```


- 查看训练结果
```
git clone --recursive https://gitlab.inria.fr/sibr/sibr_core SIBR_core
CMake GUI可直接配置成功。
打开Visual Studio 2019编译成功。

./SIBR_viewers/install/bin/SIBR_gaussianViewer_app.exe -m /path/to/output
```
