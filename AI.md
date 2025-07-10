# 目录
- [1.pytorch](##1.pytorch)
- [2.Linear-Probe-Fine-Tuning](##2.Linear-Probe-Fine-Tuning)

## 1.pytorch
- [API文档](https://docs.pytorch.org/docs/stable/index.html)
- [Deep-Learning-With-Pytorch-Tutorials](https://github.com/dragen1860/Deep-Learning-with-PyTorch-Tutorials.git)
- [NVIDA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master) 进阶学习

### 1.1. Basics
- Tensor
```
张量数据类型 torch.float, torch.double, torch.int, torch.int32, torch.int8, torch.int16, torch.int64, torch.float16;    torch.FloatTensor, torch.ByteTensor, torch.cuda.FloatTensor, a.type(), type(a), isinstance(a, torch.FloatTensor)
cpu-gpu tensors conversion 数据转化 a = a.cuda()
类型检查: isinstance(a, torch.cuda.FloatTensor)
维度Dimensions/Rank: a.size()等价于a.shape，都返回torch.Size对象

张量创建：torch.tensor(1.f); 从list创建张量 torch.tensor([1.1, 2.0]) # torch.Size([2]); 


```



## 2.Linear-Probe-Fine-Tuning
```
在深度学习中，Linear Probe（线性探测） 是一种用于评估预训练模型表征质量的评估方法，尤其在自监督学习（Self-Supervised Learning, SSL）领域广泛应用。其核心是通过冻结预训练模型参数，仅训练一个简单的线性分类器来测试模型学习特征的泛化能力。以下是详细解析：

一、核心概念与原理
定义
Linear Probe 指在预训练模型的输出层（或中间层）后添加一个线性分类器（如全连接层），仅训练该分类器而不更新预训练模型的权重45。
目的：量化预训练模型学习到的特征表示的质量。
逻辑：若预训练特征足够强，即使简单线性层也能实现高分类准确率。
操作流程
特征提取：用预训练模型（如BERT、ViT）处理输入数据，生成特征向量（冻结模型参数）。
构建线性分类器：在特征向量后接入线性层（如 nn.Linear）。
训练与评估：仅优化线性层权重，使用监督数据训练，并在测试集计算准确率（Linear Probing Accuracy）46。
二、与微调（Fine-tuning）的区别
方法	训练对象	特点	适用场景
Linear Probe	仅训练线性分类层	低成本、快速；评估特征固有质量5	预训练模型表征能力评测
Fine-tuning	更新整个模型（或部分层）	更高精度；依赖任务适配4	下游任务实际部署
```
