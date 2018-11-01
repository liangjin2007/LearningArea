# [Wiki](https://github.com/BVLC/caffe/wiki)
- Tutorials
- Model Zoo

# [DIY Deep Learning for Vision](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.g129385c8da_651_158)

- Examples and Application
   - Scene Recognition
   - Visual Style Recognition
   - Object Detection
   - Pixelwise Prediction
   - Visual Sequence Task
   - RNN, LRCN
      - video
      - language
      - dynamics
   - Deep Visuomotor Control
      - activity recognition
      - image captioning
      - video captioning
   - Facebook
      - uploaded photos run through Caffe
      - objectionable content detection
      - return back to community https://github.com/facebook/fb-caffe-exts
   - Pinterest
      - uploaded photos run through Caffe
      - deep learning for visual search
   - Adobe
      - training networks for research in vision and graphics
      - custom inference in products, including Photoshop
   - Yahoo
      - arrange user photo albums
      - curate news and restaurant photos for recommendation
      
- Net
   - Forward/Backward
   - DAG
   - Layer Protocol
      - Setup
      - Forward
      - Backward
      - Reshape
   - Blob
      - top
      - bottom
   - Loss
      - classification
         - SoftmaxWithLoss
         - HingeLoss
      - linear regression
         - EuclideanLoss
      - attributes/multiclassification
         - SigmoidCrossEntropyLoss
   - Protobuf Model Format
      - Strongly typed format
      - Auto-generates code
      - Developed by Google
      - Defines Net / Layer / Solver
      - schemas in caffe.proto
      
- Net Definition and Solver Definition
   - Convert the data to Caffe-format
      lmdb, leveldb, hdf5 / .mat, list of images, etc.
   - Define the Net
      - net.prototxt
      ```
      name: "LeNet"
      layer {
        name: "mnist"
        type: "Data"
        top: "data"
        top: "label"
        include {
          phase: TRAIN
        }
        transform_param {
          scale: 0.00390625
        }
        data_param {
          source: "examples/mnist/mnist_train_lmdb"
          batch_size: 64
          backend: LMDB
        }
      }
      layer {
        name: "mnist"
        type: "Data"
        top: "data"
        top: "label"
        include {
          phase: TEST
        }
        transform_param {
          scale: 0.00390625
        }
        data_param {
          source: "examples/mnist/mnist_test_lmdb"
          batch_size: 100
          backend: LMDB
        }
      }
      layer {
        name: "conv1"
        type: "Convolution"
        bottom: "data"
        top: "conv1"
        param {
          lr_mult: 1
        }
        param {
          lr_mult: 2
        }
        convolution_param {
          num_output: 20
          kernel_size: 5
          stride: 1
          weight_filler {
            type: "xavier"
          }
          bias_filler {
            type: "constant"
          }
        }
      }
      layer {
        name: "pool1"
        type: "Pooling"
        bottom: "conv1"
        top: "pool1"
        pooling_param {
          pool: MAX
          kernel_size: 2
          stride: 2
        }
      }
      layer {
        name: "conv2"
        type: "Convolution"
        bottom: "pool1"
        top: "conv2"
        param {
          lr_mult: 1
        }
        param {
          lr_mult: 2
        }
        convolution_param {
          num_output: 50
          kernel_size: 5
          stride: 1
          weight_filler {
            type: "xavier"
          }
          bias_filler {
            type: "constant"
          }
        }
      }
      layer {
        name: "pool2"
        type: "Pooling"
        bottom: "conv2"
        top: "pool2"
        pooling_param {
          pool: MAX
          kernel_size: 2
          stride: 2
        }
      }
      layer {
        name: "ip1"
        type: "InnerProduct"
        bottom: "pool2"
        top: "ip1"
        param {
          lr_mult: 1
        }
        param {
          lr_mult: 2
        }
        inner_product_param {
          num_output: 500
          weight_filler {
            type: "xavier"
          }
          bias_filler {
            type: "constant"
          }
        }
      }
      layer {
        name: "relu1"
        type: "ReLU"
        bottom: "ip1"
        top: "ip1"
      }
      layer {
        name: "ip2"
        type: "InnerProduct"
        bottom: "ip1"
        top: "ip2"
        param {
          lr_mult: 1
        }
        param {
          lr_mult: 2
        }
        inner_product_param {
          num_output: 10
          weight_filler {
            type: "xavier"
          }
          bias_filler {
            type: "constant"
          }
        }
      }
      layer {
        name: "accuracy"
        type: "Accuracy"
        bottom: "ip2"
        bottom: "label"
        top: "accuracy"
        include {
          phase: TEST
        }
      }
      layer {
        name: "loss"
        type: "SoftmaxWithLoss"
        bottom: "ip2"
        bottom: "label"
        top: "loss"
      }
      ```
   - Configure the Solver
      - solver.prototxt
      ```
      # The train/test net protocol buffer definition
      net: "examples/mnist/lenet_train_test.prototxt"
      # test_iter specifies how many forward passes the test should carry out.
      # In the case of MNIST, we have test batch size 100 and 100 test iterations,
      # covering the full 10,000 testing images.
      test_iter: 100
      # Carry out testing every 500 training iterations.
      test_interval: 500
      # The base learning rate, momentum and the weight decay of the network.
      base_lr: 1.0
      lr_policy: "fixed"
      momentum: 0.95
      weight_decay: 0.0005
      # Display every 100 iterations
      display: 100
      # The maximum number of iterations
      max_iter: 10000
      # snapshot intermediate results
      snapshot: 5000
      snapshot_prefix: "examples/mnist/lenet_adadelta"
      # solver mode: CPU or GPU
      solver_mode: GPU
      type: "AdaDelta"
      delta: 1e-6
      ```
     caffe train -solver solver.prototxt -gpu 0

   - Examples are your friends
   - caffe/examples/mnist,cifar10,imagenet
   - caffe/examples/*.ipynb
   - caffe/models/*

- [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)

- Examples
   - caffe/examples/*
   - caffe/models/*

- Installation
   - CentOS 
   
   - Windows
      - caffe windows branch
      - download the windows version source code 
      - scripts\build_win.cmd
      - copy caffe/python/caffe to site_packages
      - install dependencies ninja numpy scipy protobuf==3.1.0 six scikit-image pyyaml pydotplus graphviz
      
