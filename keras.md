Keras实现复杂模型
===================================================================
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


Keras mnist_swwae.py
===================================================================
s means stacked 堆叠
MaxPooling, UpSampling2D
需要求梯度的梯度，backend需要是theano。
对这两个Layer的参数比较难理解


ImageDataGenerator的用法
===================================================================
datagen.flow(x_train, y_train, batch_size=batch_size)


Keras Antirectifier 如何自定义Layer
===================================================================
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

Keras mnist net2net用知识迁移来加速训练
===================================================================
```
算是迁移学习的一种，用更宽的卷积去套住已经训练好的权重
```


Keras mnist denoising autodecoder
===================================================================
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

Keras mnist siamese
===================================================================
```
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)
```

Keras transfer learning
===================================================================
```
layer.trainable = False
```

Keras mnist mlp multiple layer perception多层感知器
====================================================================
```
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

Keras tfrecords
====================================================================
```
cache_dir = os.path.expanduser(
    os.path.join('~', '.keras', 'datasets', 'MNIST-data'))
data = mnist.read_data_sets(cache_dir, validation_size=0)

x_train_batch, y_train_batch = tf.train.shuffle_batch(
    tensors=[data.train.images, data.train.labels.astype(np.int32)],
    batch_size=batch_size,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue,
    enqueue_many=enqueue_many,
    num_threads=8)

x_train_batch = tf.cast(x_train_batch, tf.float32)
x_train_batch = tf.reshape(x_train_batch, shape=batch_shape)

y_train_batch = tf.cast(y_train_batch, tf.int32)
y_train_batch = tf.one_hot(y_train_batch, num_classes)

x_batch_shape = x_train_batch.get_shape().as_list()
y_batch_shape = y_train_batch.get_shape().as_list()

model_input = layers.Input(tensor=x_train_batch) #这个地方直接将输入数据传入
model_output = cnn_layers(model_input)
train_model = keras.models.Model(inputs=model_input, outputs=model_output)

train_model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    target_tensors=[y_train_batch]) # 这个地方直接将label传入
train_model.summary()
x_test_batch, y_test_batch = tf.train.batch(
    tensors=[data.test.images, data.test.labels.astype(np.int32)],
    batch_size=batch_size,
    capacity=capacity,
    enqueue_many=enqueue_many,
    num_threads=8)

# Create a separate test model
# to perform validation during training
x_test_batch = tf.cast(x_test_batch, tf.float32)
x_test_batch = tf.reshape(x_test_batch, shape=batch_shape)

y_test_batch = tf.cast(y_test_batch, tf.int32)
y_test_batch = tf.one_hot(y_test_batch, num_classes)

x_test_batch_shape = x_test_batch.get_shape().as_list()
y_test_batch_shape = y_test_batch.get_shape().as_list()

test_model_input = layers.Input(tensor=x_test_batch)
test_model_output = cnn_layers(test_model_input)
test_model = keras.models.Model(inputs=test_model_input, outputs=test_model_output)

# Pass the target tensor `y_test_batch` to `compile`
# via the `target_tensors` keyword argument:
test_model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'],
                   target_tensors=[y_test_batch])

# Fit the model using data from the TFRecord data tensors.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)

train_model.fit(
    epochs=epochs,
    steps_per_epoch=int(np.ceil(data.train.num_examples / float(batch_size))),
    callbacks=[EvaluateInputTensor(test_model, steps=100)])

# Save the model weights.
train_model.save_weights('saved_wt.h5')

# Clean up the TF session.
coord.request_stop()
coord.join(threads)
K.clear_session()

# Second Session to test loading trained model without tensors
x_test = np.reshape(data.test.images, (data.test.images.shape[0], 28, 28, 1))
y_test = data.test.labels
x_test_inp = layers.Input(shape=(x_test.shape[1:]))
test_out = cnn_layers(x_test_inp)
test_model = keras.models.Model(inputs=x_test_inp, outputs=test_out)

test_model.load_weights('saved_wt.h5')
test_model.compile(optimizer='rmsprop',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
test_model.summary()

loss, acc = test_model.evaluate(x_test,
                                keras.utils.to_categorical(y_test),
                                batch_size=batch_size)
print('\nTest accuracy: {0}'.format(acc))
```

Keras sklearn wrapper
=====================================================================
```
def make_model(dense_layer_sizes, filters, kernel_size, pool_size):
    '''Creates model comprised of 2 convolutional layers followed by dense layers

    dense_layer_sizes: List of layer sizes.
        This list has one number for each layer
    filters: Number of convolutional filters in each convolutional layer
    kernel_size: Convolutional kernel size
    pool_size: Size of pooling area for max pooling
    '''

    model = Sequential()
    model.add(Conv2D(filters, kernel_size,
                     padding='valid',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size))
        model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model

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

Keras cifar10_cnn_tfaugment2d.py
=====================================================================
- tf.contrib.image.compose_transforms(*transforms)
- tf.contrib.image.transform()
- tf.contrib.image.angles_to_projective_transforms
- tf.tile
- tf.convert_to_tensor
- tf.expand_dims
- tf.where
- tf.random_uniform
- tf.less

Keras数据增强的用法
=====================================================================
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


Keras SPP(SpatialPyramidPooling)
=====================================================================
from spp.SpatialPyramidPooling import SpatialPyramidPooling
```
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dense
from spp.SpatialPyramidPooling import SpatialPyramidPooling

batch_size = 64
num_channels = 3
num_classes = 10

model = Sequential()

# uses theano ordering. Note that we leave the image size as None to allow multiple image sizes
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, None, None)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(SpatialPyramidPooling([1, 2, 4]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd')

# train on 64x64x3 images
model.fit(np.random.rand(batch_size, num_channels, 64, 64), np.zeros((batch_size, num_classes)))
# train on 32x32x3 images
model.fit(np.random.rand(batch_size, num_channels, 32, 32), np.zeros((batch_size, num_classes)))
```

Keras caption_generator 代码阅读
=====================================================================
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
    


Keras R-CNN代码阅读
=====================================================================
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


Keras 新闻分类,词向量，Embedding
=====================================================================
```
import os
import sys
import numpy as np
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten,MaxPooling1D,Conv1D
from keras.models import Model
texts = []  # list of text samples
labels_index = {}  # 由标签名称映射到ID的字典
labels = []  # list of label ids
TEXT_DATA_DIR='/news20/20_newsgroup'
GLOVE_DIR='/glove.6B'
MAX_NB_WORDS=20000
MAX_SEQUENCE_LENGTH=1000
VALIDATION_SPLIT=0.2
EMBEDDING_DIM=100

# 遍历下语料文件下的所有文件夹，获得不同类别的新闻以及对应的类别标签
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    # 文件夹名称即为类别标签
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)# 随着循环，id由0递增
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                # 打开新闻样本
                t = f.read()
                i = t.find('\n\n')
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                # 保存新闻样本的列表
                f.close()
                labels.append(label_id)
                # 保存新闻样本对应的类别ID的标签
print('Found %s texts.' % len(texts))

# 将新闻样本转换为张量张量
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
# 只保留新闻文本最常见的20000个词汇
tokenizer.fit_on_texts(texts)
# 根据文本列表更新内部词汇表
sequences = tokenizer.texts_to_sequences(texts)
# 将文本转为序列
word_index = tokenizer.word_index
# 字典，将词汇表的词汇映射为下标
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#  每个新闻文本最多保留1000个词，用这1000个词来判断新闻文本的类别
labels = to_categorical(np.asarray(labels))
# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 将新闻样本分割成一个训练集和一个验证集。
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
# 打乱新闻样本
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

# 从GloVe文件中解析出每个词和它所对应的词向量，并用字典的方式存储
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),'r',encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    # 每一行的第一个元素作为键，后面的元素全部作为值
f.close()
print('Found %s word vectors.' % len(embeddings_index))
print('Found %s unique tokens.' % len(word_index))

# 我们可以根据GloVe文件解析得到的字典生成上文所定义的词向量矩阵，第i列表示词索引为i的词的词向量。
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
# 词向量矩阵初始化为0，词汇数*词汇维度
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 词向量矩阵加载到Embedding层
embedding_layer = Embedding(len(word_index) + 1, # 字典长度，即输入数据最大下标+1
                            EMBEDDING_DIM, #全连接嵌入的维度
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# 训练一个一维的卷积网络
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=128)
# 可以通过一些正则化机制（如退出）或通过微调Embedding层来更长时间地训练更高的精度。

#     若不使用预先训练的词语: Embedding从头开始初始化我们的层，并在训练期间学习它的权重。
#     这只需要替换Embedding图层（实际上，预训练的词向量引入了外部语义信息，往往对模型很有帮助）：
#embedding_layer = Embedding(len(word_index) + 1,
                            #EMBEDDING_DIM,
                            #input_length=MAX_SEQUENCE_LENGTH)
```


Keras sklearn wrapper
=====================================================================
```
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV

model = Model(...)
my_classifier = KerasClassifier(make_model, batch_size=32)
#[32]是一个输出维度为32的Dense，[32,32]是两个输出维度为32的Dense
dense_size_candidates = [[32], [64], [32, 32], [64, 64]]
#使用sklearn的分类器接口：第一个参数是可调用的函数或类对象，第二个参数是模型参数和训练参数
my_classifier = KerasClassifier(make_model, batch_size=32)
#sklearn中的GridSearchCV函数
#说明：对估计器的指定参数值进行穷举搜索。
validator = GridSearchCV(my_classifier,
                         param_grid={'dense_layer_sizes': dense_size_candidates,
                                     # nb_epoch可用于调整，即使不是模型构建函数的参数
                                     'nb_epoch': [3, 6],
                                     'nb_filters': [8],
                                     'nb_conv': [3],
                                     'nb_pool': [2]},
                         scoring='log_loss',
                         n_jobs=1)
validator.fit(X_train, y_train)
#打印最好模型的参数
print('The parameters of the best model are: ')
print(validator.best_params_)
#返回模型
best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(X_test, y_test)
print('\n')
#返回名称和数值
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)
```


Keras Sort
=====================================================================
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

Keras_Demo笔记
=====================================================================
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
   



- 数据增强
    ```
    # 这将做预处理和实时数据增加
    datagen = ImageDataGenerator(
        featurewise_center=False,  # 在数据集上将输入平均值设置为0
        samplewise_center=False,  # 将每个样本均值设置为0
        featurewise_std_normalization=False,  # 将输入除以数据集的std
        samplewise_std_normalization=False,  # 将每个输入除以其std
        zca_whitening=False,  # 应用ZCA白化
        rotation_range=0,  # 在一个范围下随机旋转图像(degrees, 0 to 180)
        width_shift_range=0.1,  # 水平随机移位图像（总宽度的分数）
        height_shift_range=0.1,  # 随机地垂直移动图像（总高度的分数）
        horizontal_flip=True,  # 随机翻转图像
        vertical_flip=False)  # 随机翻转图像

    # 计算特征方向归一化所需的数量
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)
    ```
    ```
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = datagen.flow(
                x_val, y_val,
                batch_size=batch_size,
    )
    ```
- 解释网络中一些概念
    ```
    model.add(Convolution2D(nb_filters, kernel_size,strides=(1, 1),
                            padding='valid',
                            input_shape=input_shape))#用作第一层时，需要输入input_shape参数
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))#Dense()的前面要减少连接点，防止过拟合，故通常要Dropout层或池化层
    model.add(Flatten())#Dense()层的输入通常是2D张量，故应使用Flatten层或全局平均池化
    model.add(Dense(128))
    model.add(Activation('relu'))#Dense( )层的后面通常要加非线性化函数
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))#分类
    model.summary()
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
- 图像去噪
  - 使用autoencoder去噪

FAQ
=====================================================================
# Keras FAQ: Frequently Asked Keras Questions

- [How should I cite Keras?](#how-should-i-cite-keras)
- [How can I run Keras on GPU?](#how-can-i-run-keras-on-gpu)
- [How can I run a Keras model on multiple GPUs?](#how-can-i-run-a-keras-model-on-multiple-gpus)
- [What does "sample", "batch", "epoch" mean?](#what-does-sample-batch-epoch-mean)
- [How can I save a Keras model?](#how-can-i-save-a-keras-model)
- [Why is the training loss much higher than the testing loss?](#why-is-the-training-loss-much-higher-than-the-testing-loss)
- [How can I obtain the output of an intermediate layer?](#how-can-i-obtain-the-output-of-an-intermediate-layer)
- [How can I use Keras with datasets that don't fit in memory?](#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory)
- [How can I interrupt training when the validation loss isn't decreasing anymore?](#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore)
- [How is the validation split computed?](#how-is-the-validation-split-computed)
- [Is the data shuffled during training?](#is-the-data-shuffled-during-training)
- [How can I record the training / validation loss / accuracy at each epoch?](#how-can-i-record-the-training-validation-loss-accuracy-at-each-epoch)
- [How can I "freeze" layers?](#how-can-i-freeze-keras-layers)
- [How can I use stateful RNNs?](#how-can-i-use-stateful-rnns)
- [How can I remove a layer from a Sequential model?](#how-can-i-remove-a-layer-from-a-sequential-model)
- [How can I use pre-trained models in Keras?](#how-can-i-use-pre-trained-models-in-keras)
- [How can I use HDF5 inputs with Keras?](#how-can-i-use-hdf5-inputs-with-keras)
- [Where is the Keras configuration file stored?](#where-is-the-keras-configuration-file-stored)
- [How can I obtain reproducible results using Keras during development?](#how-can-i-obtain-reproducible-results-using-keras-during-development)

---

### How should I cite Keras?

Please cite Keras in your publications if it helps your research. Here is an example BibTeX entry:

```
@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  publisher={GitHub},
  howpublished={\url{https://github.com/fchollet/keras}},
}
```

---

### How can I run Keras on GPU?

If you are running on the **TensorFlow** or **CNTK** backends, your code will automatically run on GPU if any available GPU is detected.

If you are running on the **Theano** backend, you can use one of the following methods:

**Method 1**: use Theano flags.
```bash
THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
```

The name 'gpu' might have to be changed depending on your device's identifier (e.g. `gpu0`, `gpu1`, etc).

**Method 2**: set up your `.theanorc`: [Instructions](http://deeplearning.net/software/theano/library/config.html)

**Method 3**: manually set `theano.config.device`, `theano.config.floatX` at the beginning of your code:
```python
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
```

---

### How can I run a Keras model on multiple GPUs?

We recommend doing so using the **TensorFlow** backend. There are two ways to run a single model on multiple GPUs: **data parallelism** and **device parallelism**.

In most cases, what you need is most likely data parallelism.

#### Data parallelism

Data parallelism consists in replicating the target model once on each device, and using each replica to process a different fraction of the input data.
Keras has a built-in utility, `keras.utils.multi_gpu_model`, which can produce a data-parallel version of any model, and achieves quasi-linear speedup on up to 8 GPUs.

For more information, see the documentation for [multi_gpu_model](/utils/#multi_gpu_model). Here is a quick example:

```python
from keras.utils import multi_gpu_model

# Replicates `model` on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=256)
```

#### Device parallelism

Device parallelism consists in running different parts of a same model on different devices. It works best for models that have a parallel architecture, e.g. a model with two branches.

This can be achieved by using TensorFlow device scopes. Here is a quick example:

```python
# Model where a shared LSTM is used to encode two different sequences in parallel
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# Process the first sequence on one GPU
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(tweet_a)
# Process the next sequence on another GPU
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(tweet_b)

# Concatenate results on CPU
with tf.device_scope('/cpu:0'):
    merged_vector = keras.layers.concatenate([encoded_a, encoded_b],
                                             axis=-1)
```

---

### What does "sample", "batch", "epoch" mean?

Below are some common definitions that are necessary to know and understand to correctly utilize Keras:

- **Sample**: one element of a dataset.
  - *Example:* one image is a **sample** in a convolutional network
  - *Example:* one audio file is a **sample** for a speech recognition model
- **Batch**: a set of *N* samples. The samples in a **batch** are processed independently, in parallel. If training, a batch results in only one update to the model.
  - A **batch** generally approximates the distribution of the input data better than a single input. The larger the batch, the better the approximation; however, it is also true that the batch will take longer to processes and will still result in only one update. For inference (evaluate/predict), it is recommended to pick a batch size that is as large as you can afford without going out of memory (since larger batches will usually result in faster evaluating/prediction).
- **Epoch**: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
  - When using `evaluation_data` or `evaluation_split` with the `fit` method of Keras models, evaluation will be run at the end of every **epoch**.
  - Within Keras, there is the ability to add [callbacks](https://keras.io/callbacks/) specifically designed to be run at the end of an **epoch**. Examples of these are learning rate changes and model checkpointing (saving).

---

### How can I save a Keras model?

#### Saving/loading whole models (architecture + weights + optimizer state)

*It is not recommended to use pickle or cPickle to save a Keras model.*

You can use `model.save(filepath)` to save a Keras model into a single HDF5 file which will contain:

- the architecture of the model, allowing to re-create the model
- the weights of the model
- the training configuration (loss, optimizer)
- the state of the optimizer, allowing to resume training exactly where you left off.

You can then use `keras.models.load_model(filepath)` to reinstantiate your model.
`load_model` will also take care of compiling the model using the saved training configuration
(unless the model was never compiled in the first place).

Example:

```python
from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
```

#### Saving/loading only a model's architecture

If you only need to save the **architecture of a model**, and not its weights or its training configuration, you can do:

```python
# save as JSON
json_string = model.to_json()

# save as YAML
yaml_string = model.to_yaml()
```

The generated JSON / YAML files are human-readable and can be manually edited if needed.

You can then build a fresh model from this data:

```python
# model reconstruction from JSON:
from keras.models import model_from_json
model = model_from_json(json_string)

# model reconstruction from YAML
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
```

#### Saving/loading only a model's weights

If you need to save the **weights of a model**, you can do so in HDF5 with the code below.

Note that you will first need to install HDF5 and the Python library h5py, which do not come bundled with Keras.

```python
model.save_weights('my_model_weights.h5')
```

Assuming you have code for instantiating your model, you can then load the weights you saved into a model with the *same* architecture:

```python
model.load_weights('my_model_weights.h5')
```

If you need to load weights into a *different* architecture (with some layers in common), for instance for fine-tuning or transfer-learning, you can load weights by *layer name*:

```python
model.load_weights('my_model_weights.h5', by_name=True)
```

For example:

```python
"""
Assuming the original model looks like this:
    model = Sequential()
    model.add(Dense(2, input_dim=3, name='dense_1'))
    model.add(Dense(3, name='dense_2'))
    ...
    model.save_weights(fname)
"""

# new model
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # will be loaded
model.add(Dense(10, name='new_dense'))  # will not be loaded

# load weights from first model; will only affect the first layer, dense_1.
model.load_weights(fname, by_name=True)
```

#### Handling custom layers (or other custom objects) in saved models

If the model you want to load includes custom layers or other custom classes or functions, 
you can pass them to the loading mechanism via the `custom_objects` argument: 

```python
from keras.models import load_model
# Assuming your model includes instance of an "AttentionLayer" class
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
```

Alternatively, you can use a [custom object scope](https://keras.io/utils/#customobjectscope):

```python
from keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')
```

Custom objects handling works the same way for `load_model`, `model_from_json`, `model_from_yaml`:

```python
from keras.models import model_from_json
model = model_from_json(json_string, custom_objects={'AttentionLayer': AttentionLayer})
```

---

### Why is the training loss much higher than the testing loss?

A Keras model has two modes: training and testing. Regularization mechanisms, such as Dropout and L1/L2 weight regularization, are turned off at testing time.

Besides, the training loss is the average of the losses over each batch of training data. Because your model is changing over time, the loss over the first batches of an epoch is generally higher than over the last batches. On the other hand, the testing loss for an epoch is computed using the model as it is at the end of the epoch, resulting in a lower loss.

---

### How can I obtain the output of an intermediate layer?

One simple way is to create a new `Model` that will output the layers that you are interested in:

```python
from keras.models import Model

model = ...  # create the original model

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
```

Alternatively, you can build a Keras function that will return the output of a certain layer given a certain input, for example:

```python
from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]
```

Similarly, you could build a Theano and TensorFlow function directly.

Note that if your model has a different behavior in training and testing phase (e.g. if it uses `Dropout`, `BatchNormalization`, etc.), you will need
to pass the learning phase flag to your function:

```python
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

# output in test mode = 0
layer_output = get_3rd_layer_output([x, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([x, 1])[0]
```

---

### How can I use Keras with datasets that don't fit in memory?

You can do batch training using `model.train_on_batch(x, y)` and `model.test_on_batch(x, y)`. See the [models documentation](/models/sequential).

Alternatively, you can write a generator that yields batches of training data and use the method `model.fit_generator(data_generator, steps_per_epoch, epochs)`.

You can see batch training in action in our [CIFAR10 example](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py).

---

### How can I interrupt training when the validation loss isn't decreasing anymore?

You can use an `EarlyStopping` callback:

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

Find out more in the [callbacks documentation](/callbacks).

---

### How is the validation split computed?

If you set the `validation_split` argument in `model.fit` to e.g. 0.1, then the validation data used will be the *last 10%* of the data. If you set it to 0.25, it will be the last 25% of the data, etc. Note that the data isn't shuffled before extracting the validation split, so the validation is literally just the *last* x% of samples in the input you passed.

The same validation set is used for all epochs (within a same call to `fit`).

---

### Is the data shuffled during training?

Yes, if the `shuffle` argument in `model.fit` is set to `True` (which is the default), the training data will be randomly shuffled at each epoch.

Validation data is never shuffled.

---


### How can I record the training / validation loss / accuracy at each epoch?

The `model.fit` method returns an `History` callback, which has a `history` attribute containing the lists of successive losses and other metrics.

```python
hist = model.fit(x, y, validation_split=0.2)
print(hist.history)
```

---

### How can I "freeze" Keras layers?

To "freeze" a layer means to exclude it from training, i.e. its weights will never be updated. This is useful in the context of fine-tuning a model, or using fixed embeddings for a text input.

You can pass a `trainable` argument (boolean) to a layer constructor to set a layer to be non-trainable:

```python
frozen_layer = Dense(32, trainable=False)
```

Additionally, you can set the `trainable` property of a layer to `True` or `False` after instantiation. For this to take effect, you will need to call `compile()` on your model after modifying the `trainable` property. Here's an example:

```python
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# in the model below, the weights of `layer` will not be updated during training
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# with this model the weights of the layer will be updated during training
# (which will also affect the above model since it uses the same layer instance)
trainable_model.compile(optimizer='rmsprop', loss='mse')

frozen_model.fit(data, labels)  # this does NOT update the weights of `layer`
trainable_model.fit(data, labels)  # this updates the weights of `layer`
```

---

### How can I use stateful RNNs?

Making a RNN stateful means that the states for the samples of each batch will be reused as initial states for the samples in the next batch.

When using stateful RNNs, it is therefore assumed that:

- all batches have the same number of samples
- If `x1` and `x2` are successive batches of samples, then `x2[i]` is the follow-up sequence to `x1[i]`, for every `i`.

To use statefulness in RNNs, you need to:

- explicitly specify the batch size you are using, by passing a `batch_size` argument to the first layer in your model. E.g. `batch_size=32` for a 32-samples batch of sequences of 10 timesteps with 16 features per timestep.
- set `stateful=True` in your RNN layer(s).
- specify `shuffle=False` when calling fit().

To reset the states accumulated:

- use `model.reset_states()` to reset the states of all layers in the model
- use `layer.reset_states()` to reset the states of a specific stateful RNN layer

Example:

```python

x  # this is our input data, of shape (32, 21, 16)
# we will feed it to our model in sequences of length 10

model = Sequential()
model.add(LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# we train the network to predict the 11th timestep given the first 10:
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# the state of the network has changed. We can feed the follow-up sequences:
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# let's reset the states of the LSTM layer:
model.reset_states()

# another way to do it in this case:
model.layers[0].reset_states()
```

Notes that the methods `predict`, `fit`, `train_on_batch`, `predict_classes`, etc. will *all* update the states of the stateful layers in a model. This allows you to do not only stateful training, but also stateful prediction.

---

### How can I remove a layer from a Sequential model?

You can remove the last added layer in a Sequential model by calling `.pop()`:

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"
```

---

### How can I use pre-trained models in Keras?

Code and pre-trained weights are available for the following image classification models:

- Xception
- VGG16
- VGG19
- ResNet50
- Inception v3
- Inception-ResNet v2
- MobileNet v1

They can be imported from the module `keras.applications`:

```python
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet

model = VGG16(weights='imagenet', include_top=True)
```

For a few simple usage examples, see [the documentation for the Applications module](/applications).

For a detailed example of how to use such a pre-trained model for feature extraction or for fine-tuning, see [this blog post](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).

The VGG16 model is also the basis for several Keras example scripts:

- [Style transfer](https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py)
- [Feature visualization](https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py)
- [Deep dream](https://github.com/fchollet/keras/blob/master/examples/deep_dream.py)

---

### How can I use HDF5 inputs with Keras?

You can use the `HDF5Matrix` class from `keras.utils.io_utils`. See [the HDF5Matrix documentation](/utils/#hdf5matrix) for details.

You can also directly use a HDF5 dataset:

```python
import h5py
with h5py.File('input/file.hdf5', 'r') as f:
    x_data = f['x_data']
    model.predict(x_data)
```

---

### Where is the Keras configuration file stored?

The default directory where all Keras data is stored is:

```bash
$HOME/.keras/
```

Note that Windows users should replace `$HOME` with `%USERPROFILE%`.
In case Keras cannot create the above directory (e.g. due to permission issues), `/tmp/.keras/` is used as a backup.

The Keras configuration file is a JSON file stored at `$HOME/.keras/keras.json`. The default configuration file looks like this:

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

It contains the following fields:

- The image data format to be used as default by image processing layers and utilities (either `channels_last` or `channels_first`).
- The `epsilon` numerical fuzz factor to be used to prevent division by zero in some operations.
- The default float data type.
- The default backend. See the [backend documentation](/backend).

Likewise, cached dataset files, such as those downloaded with [`get_file()`](/utils/#get_file), are stored by default in `$HOME/.keras/datasets/`.

---

### How can I obtain reproducible results using Keras during development?

During development of a model, sometimes it is useful to be able to obtain reproducible results from run to run in order to determine if a change in performance is due to an actual model or data modification, or merely a result of a new random sample.  The below snippet of code provides an example of how to obtain reproducible results - this is geared towards a TensorFlow backend for a Python 3 environment.

```python
import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Rest of code follows ...
```


Sequential Model
====================================================================
# Getting started with the Keras Sequential model

The `Sequential` model is a linear stack of layers.

You can create a `Sequential` model by passing a list of layer instances to the constructor:

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
```

You can also simply add layers via the `.add()` method:

```python
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
```

----

## Specifying the input shape

The model needs to know what input shape it should expect. For this reason, the first layer in a `Sequential` model (and only the first, because following layers can do automatic shape inference) needs to receive information about its input shape. There are several possible ways to do this:

- Pass an `input_shape` argument to the first layer. This is a shape tuple (a tuple of integers or `None` entries, where `None` indicates that any positive integer may be expected). In `input_shape`, the batch dimension is not included.
- Some 2D layers, such as `Dense`, support the specification of their input shape via the argument `input_dim`, and some 3D temporal layers support the arguments `input_dim` and `input_length`.
- If you ever need to specify a fixed batch size for your inputs (this is useful for stateful recurrent networks), you can pass a `batch_size` argument to a layer. If you pass both `batch_size=32` and `input_shape=(6, 8)` to a layer, it will then expect every batch of inputs to have the batch shape `(32, 6, 8)`.

As such, the following snippets are strictly equivalent:
```python
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
```
```python
model = Sequential()
model.add(Dense(32, input_dim=784))
```

----

## Compilation

Before training a model, you need to configure the learning process, which is done via the `compile` method. It receives three arguments:

- An optimizer. This could be the string identifier of an existing optimizer (such as `rmsprop` or `adagrad`), or an instance of the `Optimizer` class. See: [optimizers](/optimizers).
- A loss function. This is the objective that the model will try to minimize. It can be the string identifier of an existing loss function (such as `categorical_crossentropy` or `mse`), or it can be an objective function. See: [losses](/losses).
- A list of metrics. For any classification problem you will want to set this to `metrics=['accuracy']`. A metric could be the string identifier of an existing metric or a custom metric function.

```python
# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# For custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

----

## Training

Keras models are trained on Numpy arrays of input data and labels. For training a model, you will typically use the `fit` function. [Read its documentation here](/models/sequential).

```python
# For a single-input model with 2 classes (binary classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)
```

```python
# For a single-input model with 10 classes (categorical classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```

----


## Examples

Here are a few examples to get you started!

In the [examples folder](https://github.com/fchollet/keras/tree/master/examples), you will also find example models for real datasets:

- CIFAR10 small images classification: Convolutional Neural Network (CNN) with realtime data augmentation
- IMDB movie review sentiment classification: LSTM over sequences of words
- Reuters newswires topic classification: Multilayer Perceptron (MLP)
- MNIST handwritten digits classification: MLP & CNN
- Character-level text generation with LSTM

...and more.


### Multilayer Perceptron (MLP) for multi-class softmax classification:

```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```


### MLP for binary classification:

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```


### VGG-like convnet:

```python
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# Generate dummy data
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
```


### Sequence classification with LSTM:

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

### Sequence classification with 1D convolutions:

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

### Stacked LSTM for sequence classification

In this model, we stack 3 LSTM layers on top of each other,
making the model capable of learning higher-level temporal representations.

The first two LSTMs return their full output sequences, but the last one only returns
the last step in its output sequence, thus dropping the temporal dimension
(i.e. converting the input sequence into a single vector).

<img src="https://keras.io/img/regular_stacked_lstm.png" alt="stacked LSTM" style="width: 300px;"/>

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))
```


### Same stacked LSTM model, rendered "stateful"

A stateful recurrent model is one for which the internal states (memories) obtained after processing a batch
of samples are reused as initial states for the samples of the next batch. This allows to process longer sequences
while keeping computational complexity manageable.

[You can read more about stateful RNNs in the FAQ.](/getting-started/faq/#how-can-i-use-stateful-rnns)

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

# Generate dummy validation data
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_val, y_val))
```




Functional API
=======================================================================
# Getting started with the Keras functional API

The Keras functional API is the way to go for defining complex models, such as multi-output models, directed acyclic graphs, or models with shared layers.

This guide assumes that you are already familiar with the `Sequential` model.

Let's start with something simple.

-----

## First example: a densely-connected network

The `Sequential` model is probably a better choice to implement such a network, but it helps to start with something really simple.

- A layer instance is callable (on a tensor), and it returns a tensor
- Input tensor(s) and output tensor(s) can then be used to define a `Model`
- Such a model can be trained just like Keras `Sequential` models.

```python
from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # starts training
```

-----

## All models are callable, just like layers

With the functional API, it is easy to reuse trained models: you can treat any model as if it were a layer, by calling it on a tensor. Note that by calling a model you aren't just reusing the *architecture* of the model, you are also reusing its weights.

```python
x = Input(shape=(784,))
# This works, and returns the 10-way softmax we defined above.
y = model(x)
```

This can allow, for instance, to quickly create models that can process *sequences* of inputs. You could turn an image classification model into a video classification model, in just one line.

```python
from keras.layers import TimeDistributed

# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape=(20, 784))

# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)
```

-----

## Multi-input and multi-output models

Here's a good use case for the functional API: models with multiple inputs and outputs. The functional API makes it easy to manipulate a large number of intertwined datastreams.

Let's consider the following model. We seek to predict how many retweets and likes a news headline will receive on Twitter. The main input to the model will be the headline itself, as a sequence of words, but to spice things up, our model will also have an auxiliary input, receiving extra data such as the time of day when the headline was posted, etc.
The model will also be supervised via two loss functions. Using the main loss function earlier in a model is a good regularization mechanism for deep models.

Here's what our model looks like:

<img src="https://s3.amazonaws.com/keras.io/img/multi-input-multi-output-graph.png" alt="multi-input-multi-output-graph" style="width: 400px;"/>

Let's implement it with the functional API.

The main input will receive the headline, as a sequence of integers (each integer encodes a word).
The integers will be between 1 and 10,000 (a vocabulary of 10,000 words) and the sequences will be 100 words long.

```python
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)
```

Here we insert the auxiliary loss, allowing the LSTM and Embedding layer to be trained smoothly even though the main loss will be much higher in the model.

```python
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
```

At this point, we feed into the model our auxiliary input data by concatenating it with the LSTM output:

```python
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
```

This defines a model with two inputs and two outputs:

```python
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
```

We compile the model and assign a weight of 0.2 to the auxiliary loss.
To specify different `loss_weights` or `loss` for each different output, you can use a list or a dictionary.
Here we pass a single loss as the `loss` argument, so the same loss will be used on all outputs.

```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])
```

We can train the model by passing it lists of input arrays and target arrays:

```python
model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)
```

Since our inputs and outputs are named (we passed them a "name" argument),
We could also have compiled the model via:

```python
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# And trained it via:
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)
```

-----

## Shared layers

Another good use for the functional API are models that use shared layers. Let's take a look at shared layers.

Let's consider a dataset of tweets. We want to build a model that can tell whether two tweets are from the same person or not (this can allow us to compare users by the similarity of their tweets, for instance).

One way to achieve this is to build a model that encodes two tweets into two vectors, concatenates the vectors and then adds a logistic regression; this outputs a probability that the two tweets share the same author. The model would then be trained on positive tweet pairs and negative tweet pairs.

Because the problem is symmetric, the mechanism that encodes the first tweet should be reused (weights and all) to encode the second tweet. Here we use a shared LSTM layer to encode the tweets.

Let's build this with the functional API. We will take as input for a tweet a binary matrix of shape `(140, 256)`, i.e. a sequence of 140 vectors of size 256, where each dimension in the 256-dimensional vector encodes the presence/absence of a character (out of an alphabet of 256 frequent characters).

```python
import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))
```

To share a layer across different inputs, simply instantiate the layer once, then call it on as many inputs as you want:

```python
# This layer can take as input a matrix
# and will return a vector of size 64
shared_lstm = LSTM(64)

# When we reuse the same layer instance
# multiple times, the weights of the layer
# are also being reused
# (it is effectively *the same* layer)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# We can then concatenate the two vectors:
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# And add a logistic regression on top
predictions = Dense(1, activation='sigmoid')(merged_vector)

# We define a trainable model linking the
# tweet inputs to the predictions
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)
```

Let's pause to take a look at how to read the shared layer's output or output shape.

-----

## The concept of layer "node"

Whenever you are calling a layer on some input, you are creating a new tensor (the output of the layer), and you are adding a "node" to the layer, linking the input tensor to the output tensor. When you are calling the same layer multiple times, that layer owns multiple nodes indexed as 0, 1, 2...

In previous versions of Keras, you could obtain the output tensor of a layer instance via `layer.get_output()`, or its output shape via `layer.output_shape`. You still can (except `get_output()` has been replaced by the property `output`). But what if a layer is connected to multiple inputs?

As long as a layer is only connected to one input, there is no confusion, and `.output` will return the one output of the layer:

```python
a = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)

assert lstm.output == encoded_a
```

Not so if the layer has multiple inputs:
```python
a = Input(shape=(140, 256))
b = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)
encoded_b = lstm(b)

lstm.output
```
```
>> AttributeError: Layer lstm_1 has multiple inbound nodes,
hence the notion of "layer output" is ill-defined.
Use `get_output_at(node_index)` instead.
```

Okay then. The following works:

```python
assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b
```

Simple enough, right?

The same is true for the properties `input_shape` and `output_shape`: as long as the layer has only one node, or as long as all nodes have the same input/output shape, then the notion of "layer output/input shape" is well defined, and that one shape will be returned by `layer.output_shape`/`layer.input_shape`. But if, for instance, you apply the same `Conv2D` layer to an input of shape `(32, 32, 3)`, and then to an input of shape `(64, 64, 3)`, the layer will have multiple input/output shapes, and you will have to fetch them by specifying the index of the node they belong to:

```python
a = Input(shape=(32, 32, 3))
b = Input(shape=(64, 64, 3))

conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)

# Only one input so far, the following will work:
assert conv.input_shape == (None, 32, 32, 3)

conved_b = conv(b)
# now the `.input_shape` property wouldn't work, but this does:
assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
assert conv.get_input_shape_at(1) == (None, 64, 64, 3)
```

-----

## More examples

Code examples are still the best way to get started, so here are a few more.

### Inception module

For more information about the Inception architecture, see [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842).

```python
from keras.layers import Conv2D, MaxPooling2D, Input

input_img = Input(shape=(256, 256, 3))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
```

### Residual connection on a convolution layer

For more information about residual networks, see [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).

```python
from keras.layers import Conv2D, Input

# input tensor for a 3-channel 256x256 image
x = Input(shape=(256, 256, 3))
# 3x3 conv with 3 output channels (same as input channels)
y = Conv2D(3, (3, 3), padding='same')(x)
# this returns x + y.
z = keras.layers.add([x, y])
```

### Shared vision model

This model reuses the same image-processing module on two inputs, to classify whether two MNIST digits are the same digit or different digits.

```python
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

# First, define the vision modules
digit_input = Input(shape=(27, 27, 1))
x = Conv2D(64, (3, 3))(digit_input)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out = Flatten()(x)

vision_model = Model(digit_input, out)

# Then define the tell-digits-apart model
digit_a = Input(shape=(27, 27, 1))
digit_b = Input(shape=(27, 27, 1))

# The vision model will be shared, weights and all
out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

concatenated = keras.layers.concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([digit_a, digit_b], out)
```

### Visual question answering model

This model can select the correct one-word answer when asked a natural-language question about a picture.

It works by encoding the question into a vector, encoding the image into a vector, concatenating the two, and training on top a logistic regression over some vocabulary of potential answers.

```python
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential

# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

# Now let's get a tensor with the output of our vision model:
image_input = Input(shape=(224, 224, 3))
encoded_image = vision_model(image_input)

# Next, let's define a language model to encode the question into a vector.
# Each question will be at most 100 word long,
# and we will index words as integers from 1 to 9999.
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

# Let's concatenate the question vector and the image vector:
merged = keras.layers.concatenate([encoded_question, encoded_image])

# And let's train a logistic regression over 1000 words on top:
output = Dense(1000, activation='softmax')(merged)

# This is our final model:
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# The next stage would be training this model on actual data.
```

### Video question answering model

Now that we have trained our image QA model, we can quickly turn it into a video QA model. With appropriate training, you will be able to show it a short video (e.g. 100-frame human action) and ask a natural language question about the video (e.g. "what sport is the boy playing?" -> "football").

```python
from keras.layers import TimeDistributed

video_input = Input(shape=(100, 224, 224, 3))
# This is our video encoded via the previously trained vision_model (weights are reused)
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # the output will be a sequence of vectors
encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector

# This is a model-level representation of the question encoder, reusing the same weights as before:
question_encoder = Model(inputs=question_input, outputs=encoded_question)

# Let's use it to encode the question:
video_question_input = Input(shape=(100,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# And this is our video question answering model:
merged = keras.layers.concatenate([encoded_video, encoded_video_question])
output = Dense(1000, activation='softmax')(merged)
video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output)
```
