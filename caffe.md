- [Wiki](https://github.com/BVLC/caffe/wiki)
   - Tutorials
   - Model Zoo

# [DIY Deep Learning for Vision: a Hands-On Tutorial with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.g129385c8da_651_158)

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
      
- Code Examples
   - Convert the data to Caffe-format
      lmdb, leveldb, hdf5 / .mat, list of images, etc.
   - Define the Net
   - Configure the Solver
     caffe train -solver solver.prototxt -gpu 0

   - Examples are your friends
   - caffe/examples/mnist,cifar10,imagenet
   - caffe/examples/*.ipynb
   - caffe/models/*

- [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)

- 

