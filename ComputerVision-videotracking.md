# Video Tracking

## [2011] Video Tracking : theory and practise 
### What is video tracking 什么是视频跟踪
Definition: The process of estimating over time the location of one or more objects using a camera is referred to as video tracking.
- Chanllenges: clutter(杂乱背景), change in pose， Ambient illumination, Noise, Occlusions(partial and total)
  - typical motion behavior
  - pre-existing occlusion patterns
  - high-level reasoning method
  - multi-hypothesis method
  - propagate tracking hypothesis
- Components
  - extract object features
  - target representation
  - propagation
  - track management:stragegy to manage targets appearing and disappearing.
    - target disappear: terminate trajectory
    - target birth: initialise a new trajectory
    - track loss
- Problem formulation
  - single target tracking
    - Image sequence Ik
    - trajectory xk : is the status of target, 不同的应用xk的维度可能不一样。
      - location and shape
      - appearance
      - temporal variation
  - multiple target tracking
    - Xk
    - Zk ?
  - accuracy
  - precision
  - classification
    - Manually tracking
    - Automated tracking
    - Interactive tracking(semi-automated)
### Feature extraction 特征提取
- low-level feature : color, gradient, motion
  - Color : CIE colorimetry system CIE比色法系统， 三色法原理引出XYZ是个三维向量, CIELab空间， CIELuv空间，RGB空间，YIQ，YUV, YCbCr, HSL空间，
  - Gradient and derivatives
    - Sobel
  - Laplacian
    - sensitive to noise, so LoG
  - Motion
    - 2d Motion
    - Apparent Motion
      - correspondence-vector field
      - optical-flow field
      - challenge
        - sensitive to noise and illumination variation
        - moving object must be highly textured.
        - occulsion
        - aperture
      - techniques
        - differential techniques
        - phase-based techniques
        - energy-based techniques
        - correlation-based techniques
      
- mid-level feature : edges, corners, regions
- high-level feature : objects

### Target Representation 目标表示
- Shape Representation
  - Point Approximation
  - Area Approximation : rectangle, ellipse
  - Volume Approximation
  - Articulated Model
  - Deformable Model
    - Fluid model
    - Contour model
      - Maybe suitable ? [1995]Active shape model – their training and application
    - point distribution model
- Appearance representation
  - Template
    - L1
    - L2
    - Normalized cross correlation coefficient
  - Histogram: Colour histograms, Orientation histograms, Structural histograms
  - Cope with Appearance Change
    - mixture of Gaussians
    - evolve based on a modified Kalman filter
      - model drifting
    - update strategy
    - update strategy can include a contribution from the initial model
      - Active appearance models
    - high-dimensional appearances space use PCA to map to lower-dimensional space
    
### Localisation
- Single-hypothesis localisation
  - Gradient-based trackers
    - Kanade-Lucas-Tomasi(KLT) Tracker
    - Meanshift(MS) Tracker
  - Bayes tracking
  - the Kalman filter
  
- Multiple-hypothesis localisation
  - Grid sampling
  - Particle filter
  - Hybrid methods
  
### Fusion
Use Multiple features to do tracking
- Fusion strategies
  - tracker-level fusion
  - measurement-level fusion
- Feature fusion in a Particle Filter
  - Fusion of likelihoods
  - Multi-feature resampling
  - Feature reliability
  - Temporal smoothing
  
### Multi-target management
如果对象会消失和新出现，需要自动创建target和终止taget。
- measurement validation
- data association
  - nearest neighbor, knn, kdtree
  - graph matching

## Papers
[2015]Multiple hypothesis tracking revisited
[2018]Multiple hypothesis tracking algorithm for multi-target multi-camera tracking with disjoint views

## Related Topics
- Kalman filter https://www.jianshu.com/p/2768642e3abf
  - probability distribution
  - covariance matrix 
    - 对称
  - dlib/filtering/kalman_filter.cpp
  - Multi-Target-Tracking 
  - opencv/video/tracking.hpp
    KalmanFilter.
  - samples/cpp/kalman.cpp
  - http://www.cs.unc.edu/~tracker/media/pdf/SIGGRAPH2001_CoursePack_08.pdf
  - linear system theory
- assignment problem新的帧中出现的对象对应于原来哪个对象
  - 著名的Hungarian算法，需要定义goodness function
    - 一种方法是使用手写的函数去定义goodness function, 往往不好。
    - 也可以使用机器学习的方法来定义goodness function。
      - 例子：dlib/examples/assignment_learning_ex.cpp
- Active Appearance Model简称AAM
  - 参考资料[Computer Vision - ECCV 2006](https://books.google.com/books?id=ITvZ9HDPizcC&pg=PA196&lpg=PA196&dq=person+specific+AAM&source=bl&ots=9gZ28HfIWP&sig=ACfU3U3FbLd_UOk0h0ncTnSJytWPJH2R6Q&hl=en&sa=X&ved=2ahUKEwij_vjk4sboAhXKGaYKHRxhCbsQ6AEwAXoECA8QAQ#v=onepage&q=person%20specific%20AAM&f=false)
  - [2001]AAM
  - 
- Lukas-Kanade(LK) problem
## Open Source code
- https://github.com/Smorodov/Multitarget-tracker
- AAM https://gist.github.com/kurnianggoro/74de9121e122ad0bd825176751d47ecc


