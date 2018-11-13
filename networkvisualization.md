# 可视化CNN网络

# 可视化filters

# 可视化representation
- CNN code
- t-SNE可视化
- Occlusion experiments
- Visualize activations
   - [deepvis](http://yosinski.com/deepvis)
   - To visualize the function of a specific unit in a neural network,we synthesize inputs that cause that unit to have high activation
      - Visualizing Neurons without Regularization/Priors
         - [activation maximumization](https://github.com/liangjin2007/data_liangjin/blob/master/activationmaximization.jpg?raw=true)
         - gradient ascending optimization
      - Visualizing Neurons with Regularization/Priors
         - Still not natural
      - Visualizing Neurons with Better Regularization/Priors
      - 
# 可视化特征

回顾了一下研究网络可视化的论文。

- 首先是2014年一篇Visualizing and Understanding Convolutional Networks。它可以输入一张图片，给出某个layer的feature map的可视化。
具体求法如图:

- ![feature map visualization](https://github.com/liangjin2007/data_liangjin/blob/master/featuremap_visualization.jpg?raw=true)

- 然后是看了一篇[2016]Learning Deep Features for Discriminative Localization。又称为attention map

https://github.com/metalbubble/CAM

https://github.com/raghakot/keras-vis

https://fairyonice.github.io/Visualization%20of%20Filters%20with%20Keras.html

https://docs.google.com/presentation/d/15E7NlyMkG8dAMa70i2OluprBDoz3UPyAk5ZpOiCkEqw/edit#slide=id.g306972a6f1_1_477


可视化


可视化参考资料
Visualize ConvNet Features
06_Introduction to Computer Vision and convolutional networkhttps://docs.google.com/presentation/d/15E7NlyMkG8dAMa70i2OluprBDoz3UPyAk5ZpOiCkEqw/edit#slide=id.g306972a6f1_1_114
CS231n Lecture9 Understanding and Visualizing Convolutional Neural Networks
t-SNE聚类
Visualize Filters with Keras https://fairyonice.github.io/Visualization%20of%20Filters%20with%20Keras.html
Visuaize and understanding CNN https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf


class activation map 即 salient map


开源包
keras-viz
Activation maximization
Saliency maps
Class activation maps, 2015


convnets-keras: Keras-AlexNet weights and heat map visualization
Attention Map
Dense Layer Visualization
Animated gif
Deconvnet


keras-cam
picasso-viz




The source code for the paper [2016]Learning Deep Features for Discriminative Localization
https://github.com/metalbubble/CAM
PlacesCNN http://places2.csail.mit.edu/demo.html







