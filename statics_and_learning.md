# Keras

## Keras实现多输入多输出

- 模型定义的输入输出需要等量：model definition should contain equal amount of inputs and outputs.
    - 可以添加空输入，空输出，loss使用记录（self.）的方式。

- 数据生成器格式：data generator must output [input1, input2, input3, input4], [output1, output2, output3, output4]而不是[input1,output1],[input2,output2],...

- 输出名：outputs'name can be used to determine which output metrics applying to.

- 写loss函数可以将参数设成y_true, y_pred都是one_hot，只要让多输出的输出设为定义的空的输出，比如：
```
        input_image = Input(shape=input_shape, dtype=tf.float32) # image
        
        input_target_tripletloss = Input(shape=(num_classes,), dtype=tf.float32) # an input for the 2nd loss
        output_target_tripletloss= Lambda(lambda x:x, name="tripletloss")(input_target_tripletloss)

        input_target_centerloss = Input(shape=(1,), dtype=tf.int32) # an input for the 3rd loss
        output_target_centerloss = Lambda(lambda x:tf.squeeze(tf.one_hot(tf.cast(x, tf.int32), num_classes, 1.0, 0.0, dtype=tf.float32),axis=1), name="centerloss")(input_target_centerloss)

        input_target_consistencyloss = Input(shape=(num_classes,), dtype=tf.float32) # an input for the 3rd loss
        output_target_consistencyloss = Lambda(lambda x:x, name="consistencyloss")(input_target_consistencyloss)

        self.centers = Embedding(num_classes, output_dim=fc_size)(input_target_centerloss)

        input_noise_student = GaussianNoise(FLAGS.input_noise)(input_image)
        input_noise_teacher = GaussianNoise(FLAGS.input_noise)(input_image)
       
        resnet_out_student = ResNet50(input_tensor = input_noise_student, include_top=False, input_shape=input_shape, weights=None, tag="student")
        resnet_out_teacher = ResNet50(input_tensor = input_noise_teacher, include_top=False, input_shape=input_shape, weights=None, tag="teacher")
       
        x = GlobalMaxPooling2D()(resnet_out_student.output)
        embedding_student = Dense(fc_size, activation='relu', name = "embedding_student")(x)
        x = Dropout(FLAGS.dropout)(embedding_student)   
        self.probs_student = Dense(num_classes, activation='softmax', name="probs_student")(x)
        self.normed_embedding_student =  Lambda(lambda x: K.l2_normalize(x, axis=1))(embedding_student)
         
        self.student = Model(inputs=input_image, outputs=self.probs_student)
    
        x = GlobalAveragePooling2D()(resnet_out_teacher.output)
        embedding_teacher = Dense(fc_size, activation='relu')(x)
        x = Dropout(FLAGS.dropout)(embedding_teacher)
        self.probs_teacher = Dense(num_classes, activation='softmax', name="probs_teacher")(x)
        self.teacher = Model(inputs=input_image, outputs=self.probs_teacher)
        
        if use_gpu:
            self.mean_teacher = multi_gpu_model(Model(
                   inputs=[input_image, input_target_tripletloss, input_target_centerloss, input_target_consistencyloss]
                , outputs=[self.probs_student, output_target_tripletloss, output_target_centerloss, output_target_consistencyloss]
                )
            ,self.gpu_count)
        else:
            self.mean_teacher = Model(
                inputs=[input_image, input_target_tripletloss, input_target_centerloss, input_target_consistencyloss]
                , outputs=[self.probs_student, output_target_tripletloss, output_target_centerloss, output_target_consistencyloss]
                )
```

```
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import numpy as np
import keras
main_input = Input((100,), dtype='int32', name='main_input')

x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
lstm_out = LSTM(32)(x)
aux_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
aux_input = Input((5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, aux_input])
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[main_input, aux_input], outputs=[main_output, aux_output])
model.compile(optimizer='rmsprop',
loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
loss_weights={'main_output': 1., 'aux_output': 0.3})
print(model.summary())
```

## Keras实现复杂模型

https://www.imooc.com/article/details/id/31034
```
from keras.layers import Input,Embedding,LSTM,Dense,Lambda
from keras.layers.merge import dot
from keras.models import Model
from keras import backend as K

word_size = 128
nb_features = 10000
nb_classes = 10
encode_size = 64
margin = 0.1

embedding = Embedding(nb_features,word_size)
lstm_encoder = LSTM(encode_size)

def encode(input):
    return lstm_encoder(embedding(input))

q_input = Input(shape=(None,))
a_right = Input(shape=(None,))
a_wrong = Input(shape=(None,))
q_encoded = encode(q_input)
a_right_encoded = encode(a_right)
a_wrong_encoded = encode(a_wrong)

q_encoded = Dense(encode_size)(q_encoded) #一般的做法是，直接讲问题和答案用同样的方法encode成向量后直接匹配，但我认为这是不合理的，我认为至少经过某个变换。

right_cos = dot([q_encoded,a_right_encoded], -1, normalize=True)
wrong_cos = dot([q_encoded,a_wrong_encoded], -1, normalize=True)

loss = Lambda(lambda x: K.relu(margin+x[0]-x[1]))([wrong_cos,right_cos])

model_train = Model(inputs=[q_input,a_right,a_wrong], outputs=loss)
model_q_encoder = Model(inputs=q_input, outputs=q_encoded)
model_a_encoder = Model(inputs=a_right, outputs=a_right_encoded)

model_train.compile(optimizer='adam', loss=lambda y_true,y_pred: y_pred)
model_q_encoder.compile(optimizer='adam', loss='mse')
model_a_encoder.compile(optimizer='adam', loss='mse')

model_train.fit([q,a1,a2], y, epochs=10)
#其中q,a1,a2分别是问题、正确答案、错误答案的batch，y是任意形状为(len(q),1)的矩阵
```

```
from keras.layers import Input,Conv2D, MaxPooling2D,Flatten,Dense,Embedding,Lambda
from keras.models import Model
from keras import backend as K

nb_classes = 100
feature_size = 32

input_image = Input(shape=(224,224,3))
cnn = Conv2D(10, (2,2))(input_image)
cnn = MaxPooling2D((2,2))(cnn)
cnn = Flatten()(cnn)
feature = Dense(feature_size, activation='relu')(cnn)
predict = Dense(nb_classes, activation='softmax', name='softmax')(feature) #至此，得到一个常规的softmax分类模型

input_target = Input(shape=(1,))
centers = Embedding(nb_classes, feature_size)(input_target) #Embedding层用来存放中心
l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]), 1, keepdims=True), name='l2_loss')([feature,centers])

model_train = Model(inputs=[input_image,input_target], outputs=[predict,l2_loss])
model_train.compile(optimizer='adam', loss=['sparse_categorical_crossentropy',lambda y_true,y_pred: y_pred], loss_weights=[1.,0.2], metrics={'softmax':'accuracy'})

model_predict = Model(inputs=input_image, outputs=predict)
model_predict.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_train.fit([train_images,train_targets], [train_targets,random_y], epochs=10)
#TIPS：这里用的是sparse交叉熵，这样我们直接输入整数的类别编号作为目标，而不用转成one hot形式。所以Embedding层的输入，跟softmax的目标，都是train_targets，都是类别编号，而random_y是任意形状为(len(train_images),1)的矩阵。
```


## Keras pretrained_word_embeddings.py

- 文本处理之词向量
    - word2vec
    - GloVe
- Keras如何表示词向量
    - embedding_matrix
    - one_hot独日编码

- https://blog.csdn.net/sinat_22510827/article/details/90727435

## Keras mnist_acgan.py
- 可以做分类的GAN
- 自定义训练循环





## Keras mnist_siamese.py
两个点：1.loss数与ouput数一致 2. 如果有两个输入

```
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)
```


## Keras mnist_swwae.py
s means stacked 堆叠
MaxPooling, UpSampling2D
需要求梯度的梯度，backend需要是theano。
对这两个Layer的参数比较难理解


## ImageDataGenerator的用法
datagen.flow(x_train, y_train, batch_size=batch_size)


## Keras Antirectifier 如何自定义Layer
We need to specify two methods: `compute_output_shape` and `call`.
```
class Antirectifier(layers.Layer):
    '''This is the combination of a sample-wise
    L2 normalization with the concatenation of the
    positive part of the input with the negative part
    of the input. The result is a tensor of samples that are
    twice as large as the input samples.

    It can be used in place of a ReLU.

    # Input shape
        2D tensor of shape (samples, n)

    # Output shape
        2D tensor of shape (samples, 2*n)

    # Theoretical justification
        When applying ReLU, assuming that the distribution
        of the previous output is approximately centered around 0.,
        you are discarding half of your input. This is inefficient.

        Antirectifier allows to return all-positive outputs like ReLU,
        without discarding any data.

        Tests on MNIST show that Antirectifier allows to train networks
        with twice less parameters yet with comparable
        classification accuracy as an equivalent ReLU-based network.
    '''

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] *= 2
        return tuple(shape)

    def call(self, inputs):
        inputs -= K.mean(inputs, axis=1, keepdims=True)
        inputs = K.l2_normalize(inputs, axis=1)
        pos = K.relu(inputs)
        neg = K.relu(-inputs)
        return K.concatenate([pos, neg], axis=1)
```

## Keras mnist net2net用知识迁移来加速训练
```
算是迁移学习的一种，用更宽的卷积去套住已经训练好的权重
```


## Keras mnist denoising autodecoder
```
'''Trains a denoising autoencoder on MNIST dataset.

Denoising is one of the classic applications of autoencoders.
The denoising process removes unwanted noise that corrupted the
true signal.

Noise + Data ---> Denoising Autoencoder ---> Data

Given a training dataset of corrupted data as input and
true signal as output, a denoising autoencoder can recover the
hidden structure to generate clean data.

This example has modular design. The encoder, decoder and autoencoder
are 3 models that share weights. For example, after training the
autoencoder, the encoder can be used to  generate latent vectors
of input data for low-dim visualization like PCA or TSNE.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(1337)

# MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Generate corrupted MNIST images by adding noise with normal dist
# centered at 0.5 and std=0.5
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Network parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
latent_dim = 16
# Encoder/Decoder number of CNN layers and filters per layer
layer_filters = [32, 64]

# Build the Autoencoder Model
# First build the Encoder Model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# Stack of Conv2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use MaxPooling2D as alternative to strides>1
# - faster but not as good as strides>1
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)

# Shape info needed to build Decoder Model
shape = K.int_shape(x)

# Generate the latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# Instantiate Encoder Model
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# Build the Decoder Model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# Stack of Transposed Conv2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use UpSampling2D as alternative to strides>1
# - faster but not as good as strides>1
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

x = Conv2DTranspose(filters=1,
                    kernel_size=kernel_size,
                    padding='same')(x)

outputs = Activation('sigmoid', name='decoder_output')(x)

# Instantiate Decoder Model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Autoencoder = Encoder + Decoder
# Instantiate Autoencoder Model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

autoencoder.compile(loss='mse', optimizer='adam')

# Train the autoencoder
autoencoder.fit(x_train_noisy,
                x_train,
                validation_data=(x_test_noisy, x_test),
                epochs=30,
                batch_size=batch_size)

# Predict the Autoencoder output from corrupted test images
x_decoded = autoencoder.predict(x_test_noisy)

# Display the 1st 8 corrupted and denoised images
rows, cols = 10, 30
num = rows * cols
imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()

```

## Keras transfer learning
```
layer.trainable = False
```

## Keras mnist mlp multiple layer perception多层感知器

## Keras tfrecords

## Keras sklearn wrapper
```
dense_size_candidates = [[32], [64], [32, 32], [64, 64]]
my_classifier = KerasClassifier(make_model, batch_size=32)
validator = GridSearchCV(my_classifier,
                         param_grid={'dense_layer_sizes': dense_size_candidates,
                                     # epochs is avail for tuning even when not
                                     # an argument to model building function
                                     'epochs': [3, 6],
                                     'filters': [8],
                                     'kernel_size': [3],
                                     'pool_size': [2]},
                         scoring='neg_log_loss',
                         n_jobs=1)
validator.fit(x_train, y_train)

print('The parameters of the best model are: ')
print(validator.best_params_)

# validator.best_estimator_ returns sklearn-wrapped version of best model.
# validator.best_estimator_.model returns the (unwrapped) keras model
best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(x_test, y_test)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)
```

## Keras cifar10_cnn_tfaugment2d.py

- tf.contrib.image.compose_transforms(*transforms)
- tf.contrib.image.transform()
- tf.contrib.image.angles_to_projective_transforms
- tf.tile
- tf.convert_to_tensor
- tf.expand_dims
- tf.where
- tf.random_uniform
- tf.less

## Keras数据增强的用法

```
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)
```


## Keras SPP(SpatialPyramidPooling)

## Keras caption_generator 代码阅读
https://github.com/anuragmishracse/caption_generator
paper: 
- 安装
    - pip install -r requirements.txt
- 网络架构
    - ![lstm image caption](https://github.com/liangjin2007/data_liangjin/blob/master/model_lstm.png?raw=true)
    - Embedding层，只能用作模型的第一层，输入尺寸为(batch_size, sequence_length)，输出为(batch_size, sequence_length, output_dim)
- 训练数据集
Flickr_8k.trainImages.txt
```
2513260012_03d33305cf.jpg
2903617548_d3e38d7f88.jpg
3338291921_fe7ae0c8f8.jpg
488416045_1c6d903fe0.jpg
2644326817_8f45080b87.jpg
...
```
flickr_8k_train_dataset.txt
```
image_id	captions
2513260012_03d33305cf.jpg	<start> A black dog is running after a white dog in the snow . <end>
2513260012_03d33305cf.jpg	<start> Black dog chasing brown dog through snow <end>
2513260012_03d33305cf.jpg	<start> Two dogs chase each other across the snowy ground . <end>
2513260012_03d33305cf.jpg	<start> Two dogs play together in the snow . <end>
2513260012_03d33305cf.jpg	<start> Two dogs running through a low lying body of water . <end>
2903617548_d3e38d7f88.jpg	<start> A little baby plays croquet . <end>
2903617548_d3e38d7f88.jpg	<start> A little girl plays croquet next to a truck . <end>
2903617548_d3e38d7f88.jpg	<start> The child is playing croquette by the truck . <end>
2903617548_d3e38d7f88.jpg	<start> The kid is in front of a car with a put and a ball . <end>
2903617548_d3e38d7f88.jpg	<start> The little boy is playing with a croquet hammer and ball beside the car . <end>
...
```
Flickr8k.token.txt
```
1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201_693b08cb0e.jpg#1	A girl going into a wooden building .
1000268201_693b08cb0e.jpg#2	A little girl climbing into a wooden playhouse .
1000268201_693b08cb0e.jpg#3	A little girl climbing the stairs to her playhouse .
1000268201_693b08cb0e.jpg#4	A little girl in a pink dress going into a wooden cabin .
1001773457_577c3a7d70.jpg#0	A black dog and a spotted dog are fighting
1001773457_577c3a7d70.jpg#1	A black dog and a tri-colored dog playing with each other on the road .
1001773457_577c3a7d70.jpg#2	A black dog and a white dog with brown spots are staring at each other in the street .
1001773457_577c3a7d70.jpg#3	Two dogs of different breeds looking at each other on the road .
1001773457_577c3a7d70.jpg#4	Two dogs on pavement moving toward each other .
...
```
- 概念
    - 标注 描述图片的一句话
    - 训练集和测试集中每张图片可以有多个标注
    - vocab_size 单词数
    - caps 所有单词
    - max_cap_len  图像标注最多包含几个单词，也就是一句话最多多少个单词（40）。
    - 需要把单词嵌入到一个高维空间中，且保证相近到词在这个空间中比较近。
    - Embedding(vocab_size, 256, input_length=self.max_cap_len), 输入为(None，40)的整数矩阵，整数值为[0,vocab_size-1]。输出为（None，40， 256)
    - 一张图片
    - 每句标注<start>, <end>
    
## Keras R-CNN代码阅读

- 一些Python/Keras相关的使用技巧
取整，类型转化，张量与list相互转化，张量与numpy相互转化， broadcast（非常难），张量reshape（expand_dims, reshape(-1, 1)）；
将二维数组压平

- 继承Model：只需重载__init__, compile, predict三个方法
- 继承Layer: 只需重载__init__, build, call, compute_output_shape, get_config, 
- 字典如何拼接
```
def get_config(self):
    configuration = {
        "maximum_proposals": self.maximum_proposals,
        "minimum_size": self.minimum_size,
        "stride": self.stride
    }
    return {**super(ObjectProposal, self).get_config(), **configuration}
```
- broadcast
broadcast例子：形状为(1,1)的张量减去形状为（3,）的张量结果为(1, 3)的张量。
(1，1)+(3,)=> (1，3) # 左侧是更短的数组，右侧更长，所以先把短的扩充到跟长的一样长。
(1,1,1)+(3,) => (1,1,3)
(1,1,2)+(3,) => broadcast error
(15,4)+(196,1,4) ==> (196,15,4)
```
General Broadcasting Rules¶
When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions, and works its way forward. Two dimensions are compatible when
they are equal, or
one of them is 1
If these conditions are not met, a ValueError: operands could not be broadcast together exception is thrown, indicating that the arrays have incompatible shapes. The size of the resulting array is the maximum size along each dimension of the input arrays.
```

- 基于Faster RCNN实现
- 训练集数据格式
- generator如何写
- 概念
    - boxes: box 1x4 , [lbx, lby, rtx, rty], lbx range in [0, 1]
    - iou threshold: 0.5, i means intersection, u means union
    - anchor: [box1， box2, box3, ...], w, h,x_ctr, y_ctr, size=w*h, size_ratios = size*ratios, ws.K.sqrt(size_ratios)
        - ratio = w/h
        - scales
        - result: (15, 4), 15 boxes each anchor. each row is lx, ly, rx, ry
    - anchor_size: 8 or 16
    - anchor_aspect_ratio
    - anchor_aspect_scale
    
- 网络输入输出及架构
    - 输入
    inputs=[target_bounding_boxes, target_categories, target_image, target_mask, target_metadata]
    - 输出
    outputs=[output_bounding_boxes, output_categories] # batch_size* [maximum_proposals*4, maximum_proposals]
    - VGG
    - 3x3: Conv(64) 输出(None, 14, 14, 64)
    - deltas1: Conv 3x3 (None, 14, 14, 32)
    - scores1: Conv 1x1 (None, 14, 14, 9)
    - target_bounding_boxes: Input (None, None, 4)
    - target_metadata: Input (None, 3), None=>1 # 记录原图尺寸
    - 
    - Anchor层
        - 通过类似于numpy的操作，产生候补boundingboxes
        - 判断inside
        - 根据metadata及padding信息裁剪boundingboxes
        - 根据anchors, indices_inside, targets得到与anchors重叠的gt boxes
            - 用到了类似于numpy的broadcast
        -
    - RPN层
        - 只是添加了两个loss函数
    - ObjectProposal
    - 训练
    - 预测
        - target_bounding_boxes默认值形状: (image_count, 1, 4), 值为0
        - target_categories默认值形状： (image_count, 1, n_categories)， 值为0
        - target_mask默认值形状： (1, 1, 28, 28)，值为0
        - target_metadata默认值形状: 形状为(1, 3)值为[[224, 224, 1.0]]
    - 


## Keras 新闻分类,词向量，Embedding

## Keras Sort
```
K.arange, 
K.expand_dims
K.stack
K.shape
tf.argsort(values)
tf.gather(values, tf.argsort(values))

def spearman_loss(y_true, y_pred):
    n = K.shape(y_pred)[0]
    c = tf.contrib.framework.argsort(y_true[:, 0])
    d = tf.contrib.framework.argsort(y_pred[:, 0])
    return 1.0 - (K.sum(K.square(c-d))*6.0/(K.pow(n, 3)-n))
```

## Keras_Demo笔记
- GAN and VAE
- [](https://spaces.ac.cn/archives/5253/comment-page-1)
- VGG16
    ```
    from keras.applications.vgg16 import preprocess_input,decode_predictions
    model = VGG16(include_top=True, weights='imagenet')
    preds = model.predict(x)
    # 将结果解码成一个元组列表（类、描述、概率）（批次中每个样本的一个这样的列表）
    print('Predicted:', decode_predictions(preds, top=3)[0])
    ```
- utils
    ```
    from keras.utils import np_utils

    np_utils.to_categorical(y_train, nb_classes)

    from keras.utils.vis_utils import plot_model
    ```
- datasets
    ```
    from keras.datasets import mnist
    from keras.datasets import cifar10

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    ```
- network
    ```
    model.add(Flatten()) 可以把多维结构压平成一维
    model.add(Activation('relu')) 激活可以是单独一层
    ```

- Imagenet prediction to class and score
    - Various sorting
    ```
    CLASS_INDEX = None
    CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
    fpath = get_file('imagenet_class_index.json',
                     CLASS_INDEX_PATH,
                     cache_subdir='models',
                     file_hash='c2c37ea517e94d9795004a39431a14cb')
    CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    ```
- numpy方法归一化

    ```
    x -= x.mean()
    x /= (x.std()+1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    ```

- 用Callback画acc_loss曲线
    ```
    #写一个LossHistory类，保存loss和acc
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = {'batch': [], 'epoch': []}
            self.accuracy = {'batch': [], 'epoch': []}
            self.val_loss = {'batch': [], 'epoch': []}
            self.val_acc = {'batch': [], 'epoch': []}

        def on_batch_end(self, batch, logs={}):
            self.losses['batch'].append(logs.get('loss'))
            self.accuracy['batch'].append(logs.get('acc'))
            self.val_loss['batch'].append(logs.get('val_loss'))
            self.val_acc['batch'].append(logs.get('val_acc'))

        def on_epoch_end(self, batch, logs={}):
            self.losses['epoch'].append(logs.get('loss'))
            self.accuracy['epoch'].append(logs.get('acc'))
            self.val_loss['epoch'].append(logs.get('val_loss'))
            self.val_acc['epoch'].append(logs.get('val_acc'))

        def loss_plot(self, loss_type):
            iters = range(len(self.losses[loss_type]))
            #创建一个图
            plt.figure()
            # acc
            plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')#plt.plot(x,y)，这个将数据画成曲线
            # loss
            plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
            if loss_type == 'epoch':
                # val_acc
                plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
                # val_loss
                plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            plt.grid(True)#设置网格形式
            plt.xlabel(loss_type)
            plt.ylabel('acc-loss')#给x，y轴加注释
            plt.legend(loc="upper right")#设置图例显示位置
            plt.show()
    ```

- visualize convolution filter
   - 如何将一个网络中的某卷积层作为loss
   - 使滤波器的激活最大化，输入图像就是要可视化的滤波器 ？ 
   - 构建损失函数，计算输入图像的梯度。损失函数
   - 如何自己写梯度更新
   - 如何对多维数据根据某一维排序
   - 如何把多个小图片堆积到一起显示
    ```
    #说明：通过在输入空间的梯度上升，可视化VGG16的滤波器。
    from scipy.misc import imsave
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from keras.applications import vgg16
    from keras import backend as K
    img_width = 128
    img_height = 128
    layer_name = 'block5_conv1'

    #将张量转换成有效图像
    def deprocess_image(x):
        # 对张量进行规范化
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1
        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)
        # 转化到RGB数组
        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')
        return x
    def normalize(x):
        # 效用函数通过其L2范数标准化张量
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    model = vgg16.VGG16(weights='imagenet', include_top=False)
    print('Model loaded.')
    model.summary()

    input_img = model.input
    #用一个字典layer_dict存放Vgg16模型的每一层
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    print(layer_dict.keys())
    print(layer_dict.values())

    #提取滤波器
    #通过梯度上升，修改输入图像，使得滤波器的激活最大化。这是的输入图像就是我们要可视化的滤波器。
    
    kept_filters = []
    for filter_index in range(0, 200):
        # 我们只扫描前200个滤波器，
        # 但实际上有512个
        print('Processing filter %d' % filter_index)
        start_time = time.time()
        # 我们构建一个损耗函数，使所考虑的层的第n个滤波器的激活最大化
        #由字典索引输出
        layer_output = layer_dict[layer_name].output
        loss = K.mean(layer_output[:, :, :, filter_index])

        # 我们计算输入图像的梯度与这个损失
        grads = K.gradients(loss, input_img)[0]

        # 归一化技巧：我们规范化梯度
        grads = normalize(grads)

        # 此函数返回给定输入图像的损耗和梯度
        # inputs: List of placeholder tensors.
        # outputs: List of output tensors.
        iterate = K.function([input_img], [loss, grads])

        # 梯度上升的步长
        step = 1.

        # 我们从带有一些随机噪声的灰色图像开始
        input_img_data = np.random.random((1, img_width, img_height, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # 我们运行梯度上升20步
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            #每次上升一步，逐次上升
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # 一些过滤器陷入0，我们可以跳过它们
                break

        # 解码所得到的输入图像
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            #将解码后的图像和损耗值加入列表
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))


    # 我们将在8 x 8网格上选择最好的64个滤波器。
    n = 8
    # 具有最高损失的过滤器被假定为更好看。
    # 我们将只保留前64个过滤器。
    #Lambda:本函数用以对上一层的输出施以任何Theano/TensorFlow表达式
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # 构建一张黑色的图片，有足够的空间
    # 我们的尺寸为128 x 128的8 x 8过滤器，中间有5px的边距
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # 使用我们保存的过滤器填充图片
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    #显示滤波器
    plt.imshow(img)
    plt.show()
    plt.imshow(stitched_filters)
    plt.show()
    # 保存结果,将数组保存为图像
    imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
    ```
- LSTM使用，长短期记忆，一种特殊的循环神经网络RNN
    - 将图像高度作为time
    - 知乎上的图解 https://zhuanlan.zhihu.com/p/32085405
    ```
    nb_lstm_outputs = 30
    nb_time_steps = 28
    dim_input_vector - 28
    input_shape = (nb_time_steps, dim_input_vector)
    model.add(LSTM(nb_lstm_outputs, input_shape=input_shape))
    model.add(Dense(nb_classes, activation='softmax', kernel_initializer=initializers.random_normal(stddev=0.01)))
    model.summary()
    plot_model(model, to_file='lstm_model.png')
    ```

## Sequential Model
## Specifying the input shape
## Compilation
## Training
## Functional API
## All models are callable, just like layers
## Multi-input and multi-output models
## Shared layers
## The concept of layer "node"

