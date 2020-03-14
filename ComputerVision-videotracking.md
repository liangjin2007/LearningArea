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
    
## Papers
[2015]Multiple hypothesis tracking revisited
[2018]Multiple hypothesis tracking algorithm for multi-target multi-camera tracking with disjoint views

## Open Source code
https://github.com/Smorodov/Multitarget-tracker
