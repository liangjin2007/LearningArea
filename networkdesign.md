# Network Design

# Python Numpy Tutorial
- [Stanford CS231n](https://cs231n.github.io/python-numpy-tutorial/)
   - Python
      - Basic data types
         - Numbers, String, Booleans
      - Container
      - Function
      - Class
   - Numpy
      [array manipulation](https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html)
      [array math](https://docs.scipy.org/doc/numpy/reference/routines.math.html)
      - Arrays
         - np.array
         - np.zeros
         - np.ones
         - np.full
         - np.eye
         - np.arange
      - Slicing
         - a[:2] first two rows
         - a[:2, 1:3] forst two rows and columns 1 and 2
         - a[2,2] = 3, modify one element
         - a[:, 1] column 1
       - Index Array Indexing
         - example
         ```
         a = np.array([[1,2], [3, 4], [5, 6]])
         # An example of integer array indexing.
         # The returned array will have shape (3,) and
         print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

         # The above example of integer array indexing is equivalent to this:
         print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

         # When using integer array indexing, you can reuse the same
         # element from the source array:
         print(a[[0, 0], [1, 1]])  # Prints "[2 2]"
         
         b = np.array([0, 2, 0, 1])

         # Select one element from each row of a using the indices in b
         print(a[np.arange(4), b])
         ```
      - Boolean Array Indexing
         - example
         ```
         a = np.array([[1,2], [3, 4], [5, 6]])
         bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
                              # this returns a numpy array of Booleans of the same
                              # shape as a, where each slot of bool_idx tells
                              # whether that element of a is > 2.
         print(bool_idx)      # Prints "[[False False]
                              #          [ True  True]
                              #          [ True  True]]"
         # We use boolean array indexing to construct a rank 1 array
         # consisting of the elements of a corresponding to the True values
         # of bool_idx
         print(a[bool_idx])  # Prints "[3 4 5 6]"

         # We can do all of the above in a single concise statement:
         print(a[a > 2])     # Prints "[3 4 5 6]"
         ```
      - Datatypes
         - a.dtype
      - Array Math
         - np.add
         - np.substract
         - np.divide
         - np.multiply
         - np.sqrt
         - np.dot
         - np.sum
         - v.T
         - np.copyto
         - np.reshape(a,shape)
         - np.ravel(a)
         - ndarray.flat
         - ndarray.flatten
         - np.expand_dim
         - np.squeeze
         - np.transpose
         - np.moveaxis
         - np.rollaxis
         - np.swapaxis
         - np.flatten
         - np.asarray
         - np.asscalar
         - np.concatenate
         - np.stack
         - np.column_stack
         - np.dstack
         - np.vstack
         - np.hstack
         - np.block
         - np.split
         - np.vsplit
         - np.hsplit
         - np.dsplit
         - np.delete
         - np.append
         - np.resize
         - np.trim_zeros
         - 
         - np.sin
         - np.cos
         - np.tan
         - np.arcsin
         - np.arccos
         - np.arctan
         - np.hydot
         - np.arctan2
         - np.degrees
         - np.radians
         - np.unwrap
         - np.deg2rad
         - np.rad2deg
         - np.sinh
         - np.cosh
         - np....
         - 
         - np.around
         - np.fix
         - np.rint
         - np.floor
         - np.ceil
         - np.trunc
         - 
         - np.prod
         - np.nanprod
         - np.sum
         - np.nansum
         - np.cumsum
         - np.cumprod
         - np.nancumprod
         - np.nancumsum
         - np.diff
         - np.ediff1d
         - np.gradient
         - np.cross
         - np.trapz
         - 
         - log, exp, log2, log10, log1p, logaddexp, logaddexp2, exp2, expm1, exp2
         - 
         - np.signbit
         - np.copysign
         - 
         - positive, negative, add, substract, ...
         -
         - maximum
         - minimum
         
      - Broadcasting
         - large array and small array
         - Broadcasting two arrays together follows these rules:

            - If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
            - The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.
            - The arrays can be broadcast together if they are compatible in all dimensions.
            - After broadcasting, each array behaves as if it had shape equal to the elementwise maximum of shapes of the two input arrays.
            - In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension
            
       
# Tensorflow Setup Detail
- On MacOSX, No GPU Version
- Only Support CUDA8.0 and cuDNN6
- On Windows, Only Support Python3.5 and Python3.6

# Course
- stanford-tensorflow-tutorials on github
   - [github](https://github.com/chiphuyen/stanford-tensorflow-tutorials)
   - [cs20si](https://web.stanford.edu/class/cs20si/syllabus.html)
   - Data Flow Graph
      - A TF program often has 2 phases: 
         - Assemble a graph 
         - Use a session to execute operations in the graph.
      - Tensor
      - Nodes: operations, variables, constants
      - Edges: Tensor
      ```
      a = tf.add(3, 5)
      print(a) # >> Tensor("Add:0", shape=(), dtype=int32)
      ```
      - session
      - subgraph
      - GPU
      - Distributed Computation
         - with tf.device('/gpu:2'):
         - sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
      - default graph
         - g = tf.get_default_graph()
      - multiple graph
      ```
      g = tf.Graph()
      with g.as_default():
         x = tf.add(3, 5)
      sess = tf.Session(graph=g)
      with tf.Session() as sess:
         sess.run(x)
      ```
   - Tensorflow Ops
      - Warning Level
         - os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
      - tensorboard visualization
         - ![节点符号表](https://github.com/liangjin2007/data_liangjin/blob/master/tensorboard.jpg?raw=true)
         - [介绍](https://blog.csdn.net/lqfarmer/article/details/77239504)
         - [TensorBoard: Graph Visualization](https://www.tensorflow.org/guide/graph_viz#interaction)
         - writer = tf.summary.FileWriter('./graphs/l2', sess.graph) writer.close()
         - tensorboard --logdir='./graphs/l2'
         - 数据连接
         - 控制连接
      - Constants, Sequences, Variables, Ops
         - initializer
            - tf.zeros()
            - tf.zeros_like()
            - tf.ones()
            - tf.ones_like
            - tf.fill
            - tf.lin_space
            - tf.range
            - tf.set_random_seed(seed)
            -tf.random_normal
               tf.truncated_normal
               tf.random_uniform
               tf.random_shuffle
               tf.random_crop
               tf.multinomial
               tf.random_gamma
         - ![operations](https://github.com/liangjin2007/data_liangjin/blob/master/operations.jpg?raw=true)
         - Arithmetic Ops
            - abs, negative, square, round, sqrt, rsqrt, pow, exp
         - Div
            ```
            a = tf.constant([2, 2], name='a')
            b = tf.constant([[0, 1], [2, 3]], name='b')
            with tf.Session() as sess:
               print(sess.run(tf.div(b, a)))             ⇒ [[0 0] [1 1]]
               print(sess.run(tf.divide(b, a)))          ⇒ [[0. 0.5] [1. 1.5]]
               print(sess.run(tf.truediv(b, a)))         ⇒ [[0. 0.5] [1. 1.5]]
               print(sess.run(tf.floordiv(b, a)))        ⇒ [[0 0] [1 1]]
               print(sess.run(tf.realdiv(b, a)))         ⇒ # Error: only works for real values
               print(sess.run(tf.truncatediv(b, a)))     ⇒ [[0 0] [1 1]]
               print(sess.run(tf.floor_div(b, a)))       ⇒ [[0 0] [1 1]]
            ```
         - Data type
            ```
            t_0 = 19 			         			# scalars are treated like 0-d tensors
            tf.zeros_like(t_0)                  			# ==> 0
            tf.ones_like(t_0)                    			# ==> 1

            t_1 = [b"apple", b"peach", b"grape"] 	# 1-d arrays are treated like 1-d tensors
            tf.zeros_like(t_1)                   			# ==> [b'' b'' b'']
            tf.ones_like(t_1)                    			# ==> TypeError: Expected string, got 1 of type 'int' instead.

            t_2 = [[True, False, False],
              [False, False, True],
              [False, True, False]]         		# 2-d arrays are treated like 2-d tensors

            tf.zeros_like(t_2)                   			# ==> 3x3 tensor, all elements are False
            tf.ones_like(t_2)                    			# ==> 3x3 tensor, all elements are True
            ```
            - tf.float16, tf.float32, tf.float64, tf.bfloat16, tf.complex64, tf.int8, tf.uint6, tf.int16, tf.uint16, tf.int32, tf.uint32, tf.int64, tf.bool, tf.string, tf.resource
            - can input numpy datatype
         - Constants are stored in the graph definition
            - tf.constant
            - If constant is big, graph def will be very big
         - print graph definition
            - print(tf.get_default_graph().as_graph_def()) 
            - print(sess.graph.as_graph_def())
         - Variable
            - dtype
            - shape
            - create variable
            ```
            s = tf.Variable(2, name='scalar') 
            m = tf.Variable([[0, 1], [2, 3]], name='matrix') 
            W = tf.Variable(tf.zeros([784,10]), name='big_matrix')
            V = tf.Variable(tf.truncated_normal([784, 10]), name='normal_matrix')

            s = tf.get_variable('scalar', initializer=tf.constant(2)) 
            m = tf.get_variable('matrix', initializer=tf.constant([[0, 1], [2, 3]]))
            W = tf.get_variable('big_matrix', shape=(784, 10), initializer=tf.zeros_initializer())
            V = tf.get_variable('normal_matrix', shape=(784, 10), initializer=tf.truncated_normal_initializer())
            ```
            - Why tf.constant but tf.Variable?
               - tf.constant is an op
               - tf.Variable is a class with many ops
               ```
               tf.Variable holds several ops:
               x = tf.Variable(...) 

               x.initializer # init op
               x.value() # read op
               x.assign(...) # write op
               x.assign_add(...) # and more

               ```
            - sess.run(tf.global_variables_initializer())
            - sess.run(tf.variables_initializer([a, b]))
            - sess.run(W.initializer)
            - eval a variable
               w.eval() similar to sess.run(W)
            - Each session maintains its own copy of variables
            ```
            W = tf.Variable(10)
            sess1 = tf.Session()
            sess2 = tf.Session()

            sess1.run(W.initializer)
            sess2.run(W.initializer)

            print(sess1.run(W.assign_add(10))) 		# >> 20
            print(sess2.run(W.assign_sub(2))) 		# >> 8

            print(sess1.run(W.assign_add(100))) 		# >> 120
            print(sess2.run(W.assign_sub(50))) 		# >> -42

            sess1.close()
            sess2.close()

            ```
            - ![节点符号表](https://github.com/liangjin2007/data_liangjin/blob/master/UnderstandOperationInTheVisualization.jpg?raw=true)
            - This graph contains three separate operations : W/Assign, W, Assign, so need to call sess.run(W.initializer)+sess.run(W) or sess.run(assign_op)+sess.run(W)
         - Control Dependencies
         ```
         tf.Graph.control_dependencies(control_inputs)
         # defines which ops should be run first
         # your graph g have 5 ops: a, b, c, d, e
         g = tf.get_default_graph()
         with g.control_dependencies([a, b, c]):
            # 'd' and 'e' will only run after 'a', 'b', and 'c' have executed.
            d = ...
            e = …

         ```
      - Placeholder
         - Assemble the graph first without knowing the values needed for computation
         
         - feed_dict with placeholder
            - a = tf.placeholder(tf.float32, shape=[3])
            - sess.run(b, {a:[1,2,3]})
         - feed_dict with variable, Feeding values to TF ops 
            - a = tf.add(2,5)
            - b = tf.multiply(a,3)
            - sess.run(b, feed_dict={a:15})
         - tf.Graph.is_feedable(tensor)
         - Extremely helpful for testing
            - Feed in dummy values to test parts of a large graph
         - lazy loading
            - Separate definition of ops from computing/running ops 
            - Use Python property to ensure function is also loaded once the first time it is called*
   
   -Basic Model in Tensorflow
      - Steps:
         - Read in data
         - Create placeholders for inputs and labels
         - Create weight and bias
         - Inference
         - Specify loss function
         - Create optimizer
         - Train our model
            - Write log files using a FileWriter
         - See it on TensorBoard
         - Plot the results with matplotlib
         - Huber loss
         ```
         def huber_loss(labels, predictions, delta=14.0):
         residual = tf.abs(labels - predictions)
         def f1(): return 0.5 * tf.square(residual)
         def f2(): return delta * residual - 0.5 * tf.square(delta)
         return tf.cond(residual < delta, f1, f2)
         ```
      - Linear Regression
         - Find a linear relationship between X and Y to predict Y from X
         - Inference: Y_predicted = w * X + b
         - Mean squared error: E[(y - y_predicted)2]
         - start.py
      - TF Control Flow
         - Control Flow Ops
            - tf.group, tf.count_up_to, tf.cond, tf.case, tf.while_loop, ...
         - Comparison Ops
            - tf.equal, tf.not_equal, tf.less, tf.greater, tf.where, ...
         - Logical Ops
            - tf.logical_and, tf.logical_not, tf.logical_or, tf.logical_xor
         - Debugging Ops
            - tf.is_finite, tf.is_inf, tf.is_nan, tf.Assert, tf.Print, ...
      - tf.data
         - Placeholder
            - Pro: put the data processing outside TensorFlow, making it easy to do in Python
            - Cons: users often end up processing their data in a single thread and creating data bottleneck that slows execution down.
         - Instead of doing inference with placeholders and feeding in data later, do inference directly with data
         - Store data in tf.data.Dataset
            - tf.data.Dataset.from_tensor_slices((features, labels))
            - tf.data.Dataset.from_generator(gen, output_types, output_shapes)
            - dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
         - Can also create Dataset from files
            - tf.data.TextLineDataset(filenames)
            - tf.data.FixedLengthRecordDataset(filenames)
            - tf.data.TFRecordDataset(filenames)
      - tf.Iterator
         - Create an iterator to iterate through samples in Dataset
         - iterator = dataset.make_one_shot_iterator()
            - Iterates through the dataset exactly once. No need to initialization.
         - iterator = dataset.make_initializable_iterator()
            - Iterates through the dataset as many times as we want. Need to initialize with each epoch.
      - Does tf.data really perform better?
         - With placeholder: 9.05271519 seconds
         - With tf.data: 6.12285947 seconds
      - Should we always use tf.data?
         - For prototyping, feed dict can be faster and easier to write (pythonic)
         - tf.data is tricky to use when you have complicated preprocessing or multiple data sources
         - NLP data is normally just a sequence of integers. In this case, transferring the data over to GPU is pretty quick, so the speedup of tf.data isn't that large
      - Optimizer
      - Session looks at all trainable variables that loss depends on and update them
      - Trainable variables
      - tf.Variable(initial_value=None, trainable=True,...)
      - Code Exampe
         ```
         import os
         #os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
         import numpy as np
         import tensorflow as tf
         import time
         import utils

         # Define paramaters for the model
         learning_rate = 0.01
         batch_size = 128
         n_epochs = 30
         n_train = 60000
         n_test = 10000
         # Step 1: Read in data
         mnist_folder = 'data/mnist'
         utils.download_mnist(mnist_folder)
         train, val, test = utils.read_mnist(mnist_folder, flatten=True)
         # Step 2: Create datasets and iterator
         train_data = tf.data.Dataset.from_tensor_slices(train)
         train_data = train_data.shuffle(10000) # if you want to shuffle your data
         train_data = train_data.batch(batch_size)
         test_data = tf.data.Dataset.from_tensor_slices(test)
         test_data = test_data.batch(batch_size)
         iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                                    train_data.output_shapes)
         img, label = iterator.get_next()
         train_init = iterator.make_initializer(train_data)	# initializer for train_data
         test_init = iterator.make_initializer(test_data)	# initializer for train_data
         # Step 3: create weights and bias
         # w is initialized to random variables with mean of 0, stddev of 0.01
         # b is initialized to 0
         # shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
         # shape of b depends on Y
         w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer(0, 0.01))
         b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())
         # Step 4: build model
         # the model that returns the logits.
         # this logits will be later passed through softmax layer
         logits = tf.matmul(img, w) + b 
         # Step 5: define loss function
         # use cross entropy of softmax of logits as the loss function
         entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
         loss = tf.reduce_mean(entropy, name='loss') # computes the mean over all the examples in the batch
         # Step 6: define training op
         # using gradient descent with learning rate of 0.01 to minimize loss
         optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
         # Step 7: calculate accuracy with test set
         preds = tf.nn.softmax(logits)
         correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
         accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
         writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
         with tf.Session() as sess:
             start_time = time.time()
             sess.run(tf.global_variables_initializer())
             # train the model n_epochs times
             for i in range(n_epochs): 	
                 sess.run(train_init)	# drawing samples from train_data
                 total_loss = 0
                 n_batches = 0
                 try:
                     while True:
                         _, l = sess.run([optimizer, loss])
                         total_loss += l
                         n_batches += 1
                 except tf.errors.OutOfRangeError:
                     pass
                 print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
             print('Total time: {0} seconds'.format(time.time() - start_time))
             # test the model
             sess.run(test_init)			# drawing samples from test_data
             total_correct_preds = 0
             try:
                 while True:
                     accuracy_batch = sess.run(accuracy)
                     total_correct_preds += accuracy_batch
             except tf.errors.OutOfRangeError:
                 pass
             print('Accuracy {0}'.format(total_correct_preds/n_test))
         writer.close()
         ```
   
   - Eager execution
      - TF 1.5
      - Similar to Numpy Python
      - Can use pdb to debug code
      - Code Example
      ```
      """ Starter code for a simple regression example using eager execution.
      Created by Akshay Agrawal (akshayka@cs.stanford.edu)
      CS20: "TensorFlow for Deep Learning Research"
      cs20.stanford.edu
      Lecture 04
      """
      import time

      import tensorflow as tf
      import tensorflow.contrib.eager as tfe
      import matplotlib.pyplot as plt

      import utils

      DATA_FILE = 'data/birth_life_2010.txt'

      # In order to use eager execution, `tfe.enable_eager_execution()` must be
      # called at the very beginning of a TensorFlow program.
      tfe.enable_eager_execution()

      # Read the data into a dataset.
      data, n_samples = utils.read_birth_life_data(DATA_FILE)
      dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))

      # Create variables.
      w = tfe.Variable(0.0)
      b = tfe.Variable(0.0)

      # Define the linear predictor.
      def prediction(x):
        return x * w + b

      # Define loss functions of the form: L(y, y_predicted)
      def squared_loss(y, y_predicted):
        return (y - y_predicted) ** 2

      def huber_loss(y, y_predicted, m=1.0):
        """Huber loss."""
        t = y - y_predicted
        # Note that enabling eager execution lets you use Python control flow and
        # specificy dynamic TensorFlow computations. Contrast this implementation
        # to the graph-construction one found in `utils`, which uses `tf.cond`.
        return t ** 2 if tf.abs(t) <= m else m * (2 * tf.abs(t) - m)

      def train(loss_fn):
        """Train a regression model evaluated using `loss_fn`."""
        print('Training; loss function: ' + loss_fn.__name__)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

        # Define the function through which to differentiate.
        def loss_for_example(x, y):
          return loss_fn(y, prediction(x))

        # `grad_fn(x_i, y_i)` returns (1) the value of `loss_for_example`
        # evaluated at `x_i`, `y_i` and (2) the gradients of any variables used in
        # calculating it.
        grad_fn = tfe.implicit_value_and_gradients(loss_for_example)

        start = time.time()
        for epoch in range(100):
          total_loss = 0.0
          for x_i, y_i in tfe.Iterator(dataset):
            loss, gradients = grad_fn(x_i, y_i)
            # Take an optimization step and update variables.
            optimizer.apply_gradients(gradients)
            total_loss += loss
          if epoch % 10 == 0:
            print('Epoch {0}: {1}'.format(epoch, total_loss / n_samples))
        print('Took: %f seconds' % (time.time() - start))
        print('Eager execution exhibits significant overhead per operation. '
              'As you increase your batch size, the impact of the overhead will '
              'become less noticeable. Eager execution is under active development: '
              'expect performance to increase substantially in the near future!')

      train(huber_loss)
      plt.plot(data[:,0], data[:,1], 'bo')
      # The `.numpy()` method of a tensor retrieves the NumPy array backing it.
      # In future versions of eager, you won't need to call `.numpy()` and will
      # instead be able to, in most cases, pass Tensors wherever NumPy arrays are
      # expected.
      plt.plot(data[:,0], data[:,0] * w.numpy() + b.numpy(), 'r',
               label="huber regression")
      plt.legend()
      plt.show()

      ```
   - Manage Experiments
      - Word Embedding in TensorFlow
         - word2vec
         - Structure Model in TF
         - Model as class
         - Embedding visualization
            - t-SNE Visualization
         - grouping together nodes
            - tf.name_scope() 
      - Variable Sharing
         - Want multiple inputs to use same weights and bias
         - tf.variable_scope()
         - Variable sharing: The problem
         ```
         def two_hidden_layers(x):
             w1 = tf.Variable(tf.random_normal([100, 50]), name='h1_weights')
             b1 = tf.Variable(tf.zeros([50]), name='h1_biases')
             h1 = tf.matmul(x, w1) + b1

             w2 = tf.Variable(tf.random_normal([50, 10]), name='h2_weights')
             b2 = tf.Variable(tf.zeros([10]), name='2_biases')
             logits = tf.matmul(h1, w2) + b2
             return logits
         logits1 = two_hidden_layers(x1)
         logits2 = two_hidden_layers(x2)
         ```
         - ![result1](https://github.com/liangjin2007/data_liangjin/blob/master/two_twolayers.jpg?raw=true)
         - Two sets of variables are created.

         - Put your variables within a scope and reuse all variables within that scope
            - use tf.variable_scope('two_layers') as scope
            - scope.reuse_variables()
            ```
            def two_hidden_layers(x):
                assert x.shape.as_list() == [200, 100]
                w1 = tf.get_variable("h1_weights", [100, 50], initializer=tf.random_normal_initializer())
                b1 = tf.get_variable("h1_biases", [50], initializer=tf.constant_initializer(0.0))
                h1 = tf.matmul(x, w1) + b1
                assert h1.shape.as_list() == [200, 50]  
                w2 = tf.get_variable("h2_weights", [50, 10], initializer=tf.random_normal_initializer())
                b2 = tf.get_variable("h2_biases", [10], initializer=tf.constant_initializer(0.0))
                logits = tf.matmul(h1, w2) + b2
                return logits
            with tf.variable_scope('two_layers') as scope:
                logits1 = two_hidden_layers(x1)
                scope.reuse_variables()
                logits2 = two_hidden_layers(x2)
            ```
            - ![result2](https://github.com/liangjin2007/data_liangjin/blob/master/shared_variables1.jpg?raw=true)
         - Reusable code?
            - Fetch variables if they already exist, Else, create them
            ```
            def fully_connected(x, output_dim, scope):
                with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
                    w = tf.get_variable("weights", [x.shape[1], output_dim], initializer=tf.random_normal_initializer())
                    b = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(0.0))
                    return tf.matmul(x, w) + b

            def two_hidden_layers(x):
                h1 = fully_connected(x, 50, 'h1')
                h2 = fully_connected(h1, 10, 'h2')

            with tf.variable_scope('two_layers') as scope:
                logits1 = two_hidden_layers(x1)
                logits2 = two_hidden_layers(x2)
            ```
            - ![result3](https://github.com/liangjin2007/data_liangjin/blob/master/shared_variables2.jpg?raw=true)
      - Manage Experiment
         - tf.train.Saver
            - saves graph’s variables in binary files
            - Only save variables, not graph
            - Checkpoints map variable names to tensors
            - Can also choose to save certain variables
            - You can save your variables in one of three ways:
               - v1 = tf.Variable(..., name='v1') 
               - v2 = tf.Variable(..., name='v2') 
               - saver = tf.train.Saver({'v1': v1, 'v2': v2})
               - saver = tf.train.Saver([v1, v2])
               - saver = tf.train.Saver({v.op.name: v for v in [v1, v2]}) # similar to a dict

         - Saves sessions, not graphs!
            - tf.train.Saver.save(sess, save_path, global_step=None...)
            - tf.train.Saver.restore(sess, save_path)
            ```
            # define model
            model = SkipGramModel(params)

            # create a saver object
            saver = tf.train.Saver()

            with tf.Session() as sess:
               for step in range(training_steps): 
                  sess.run([optimizer])

                  # save model every 1000 steps
                  if (step + 1) % 1000 == 0:
                     saver.save(sess, 
            'checkpoint_directory/model_name',
            global_step=step)
            ```
            - global step
               - global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
               - Need to tell optimizer to increment global step, This can also help your optimizer know when to decay learning rate
               ```
               optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
               ```
         - Restore variables
            - saver.restore(sess, 'checkpoints/name_of_the_checkpoint') # need to first build graph
         - Restore the latest checkpoint
            ```
            # check if there is checkpoint
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

            # check if there is a valid checkpoint path
            if ckpt and ckpt.model_checkpoint_path:
                 saver.restore(sess, ckpt.model_checkpoint_path)
            ```
         - tf.summary()
            - summary is operation
            - Step 1: create summary
               - merge them all into one summary op to make managing them easier
               ```
               with tf.name_scope("summaries"):
                   tf.summary.scalar("loss", self.loss)
                   tf.summary.scalar("accuracy", self.accuracy)            
                   tf.summary.histogram("histogram loss", self.loss)
                   summary_op = tf.summary.merge_all()
               ```
            - Step 2: run them
               ```
               loss_batch, _, summary = sess.run([loss, 
                     optimizer, 
                     summary_op])
               ```
            - Step 3: write summary to file
               ```
               writer.add_summary(summary, global_step=step)
               ```
            - Putting it together
               ```
               tf.summary.scalar("loss", self.loss)
               tf.summary.histogram("histogram loss", self.loss)
               summary_op = tf.summary.merge_all()

               saver = tf.train.Saver() # defaults to saving all variables

               with tf.Session() as sess:
                   sess.run(tf.global_variables_initializer())
                   ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
                   if ckpt and ckpt.model_checkpoint_path:
                       saver.restore(sess, ckpt.model_checkpoint_path)

                   writer = tf.summary.FileWriter('./graphs', sess.graph)
                   for index in range(10000):
                       ...
                       loss_batch, _, summary = sess.run([loss, optimizer, summary_op])
                       writer.add_summary(summary, global_step=index)

                       if (index + 1) % 1000 == 0:
                           saver.save(sess, 'checkpoints/skip-gram', index)
               ```
            - See summaries on TensorBoard
      
      - Control Randomization
         - Op level random seed
            - my_var = tf.Variable(tf.truncated_normal((-1.0,1.0), stddev=0.1, seed=0))
         - Sessions keep track of random state
            - Each new session restarts the random state
            ```
            c = tf.random_uniform([], -10, 10, seed=2)
            with tf.Session() as sess:
               print(sess.run(c)) # >> 3.57493
               print(sess.run(c)) # >> -5.97319

            c = tf.random_uniform([], -10, 10, seed=2)

            with tf.Session() as sess:
               print(sess.run(c)) # >> 3.57493

            with tf.Session() as sess:
               print(sess.run(c)) # >> 3.57493

            ```
         - Op level seed: each op keeps its own seed
         - Graph level seed
            - Note that the result is different from op-level seed
            ```
            tf.set_random_seed(2)
            c = tf.random_uniform([], -10, 10)
            d = tf.random_uniform([], -10, 10)

            with tf.Session() as sess:
                print(sess.run(c)) # >> -4.00752
                print(sess.run(d)) # >> -2.98339

            ```
      - Autodiff
         - Where are the gradients?
         - TensorFlow builds the backward path for you!
         - tf.gradients(y, [xs])
            - Take derivative of y with respect to each tensor in the list [xs]
            ```
            x = tf.Variable(2.0)
            y = 2.0 * (x ** 3)
            z = 3.0 + y ** 2
            grad_z = tf.gradients(z, [x, y])
            with tf.Session() as sess:
               sess.run(x.initializer)
               print(sess.run(grad_z)) # >> [768.0, 32.0]
            # 768 is the gradient of z with respect to x, 32 with respect to y

            ```
         - Gradient Computation
            ```
            tf.gradients(ys, xs, grad_ys=None, ...)
            tf.stop_gradient(input, name=None)
            # prevents the contribution of its inputs to be taken into account
            tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)
            tf.clip_by_norm(t, clip_norm, axes=None, name=None)
            ```
         - Vanishing/exploding gradients
   - Convolutional Network
      - convolution layer
         - tf.layers.conv2d
         - ![formula](https://github.com/liangjin2007/data_liangjin/blob/master/convolutionlayer.jpg?raw=true)
      - pooling layer
         - ![formula](https://github.com/liangjin2007/data_liangjin/blob/master/poolinglayer.jpg?raw=true)
      - Visualize ConvNet Features
         - Look at filters
         - First layers: networks learn similar features
         - Last layers：Nearest Neighbor in 4096d feature spaces
         - Last layers: Dimensionality Reduction
            - t-SNE
         - (guided) backprop
            - How does the chosen neuron respond to the image?
            - Step 1: Feed image into net
            - Step 2: Set gradient of chosen layer to all zero, except 1 for the chosen neuron
            - Step 3: Backprop to image
            - 
         - Gradient ascent
            - Generate a synthetic image that maximally activates a neuron
            - I* = arg maxI f(I) + R(I)
            - Step 1: Initialize image to zeros
            - Step 2: Repeat:
               - 2. Forward image to compute current scores
               - 3. Set gradient of scores to be 1 for target class, 0 for others
               - 4. Backprop to get gradient on image
               - 5. Make a small update to the image

   - Convolution Network in TF
      - tf.nn.conv2d
      - tf.nn.max_pool
      - fully connected
         - fc = tf.matmul(pool2, w) + b
      - softmax
         - Loss function
            - tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
         - Predict
            - tf.nn.softmax(logits_batch)
      - tf.layers
         - tf.layers.conv2d
         - tf.layers.max_pooling2d
         - tf.layers.dense
         - tf.layers.dropout
   - Style Transfer
      - TFRecord
         - The recommended format for TensorFlow
         - A serialized tf.train.Example protobuf object
         - Why binary
            - make better use of disk cache
            - faster to move around
            - can handle data of different types e.g. you can put both images and labels in one place
         - Convert to TFRecord format
            - Feature: an image
            - Label: a number
            ```
            # Step 1: create a writer to write tfrecord to that file
            writer = tf.python_io.TFRecordWriter(out_file)

            # Step 2: get serialized shape and values of the image
            shape, binary_image = get_image_binary(image_file)

            # Step 3: create a tf.train.Features object
            features = tf.train.Features(feature={'label': _int64_feature(label),
                                                'shape': _bytes_feature(shape),
                                                'image': _bytes_feature(binary_image)})

            # Step 4: create a sample containing of features defined above
            sample = tf.train.Example(features=features)

            # Step 5: write the sample to the tfrecord file
            writer.write(sample.SerializeToString())
            writer.close()
            def _int64_feature(value):
                return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

            def _bytes_feature(value):
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

            ```
         - Read TFRecord
            - Using tf.data.TFRecordDataset
            ```
            dataset = tf.data.TFRecordDataset(tfrecord_files)
            dataset = dataset.map(_parse_function)
            def _parse_function(tfrecord_serialized):
               features={'label': tf.FixedLenFeature([], tf.int64),
                          'shape': tf.FixedLenFeature([], tf.string),
                          'image': tf.FixedLenFeature([], tf.string)}

               parsed_features = tf.parse_single_example(tfrecord_serialized, features)

               return parsed_features['label'], parsed_features['shape'], parsed_features['image']

            ```
      - Style Transfer
   
   - PPTn
# Tools

- Visual Studio Code
   - Jupyter
      - Python extension
      - Jupyter txtension
      - Visual Studio Code Tools for AI
      - pip3 install jupyter
      - right key menu of a ipynb file
   - Markdown previewer
      - Markdown Preview Enhanced extension
      - Ctrl+K, released, V
   - Tensorflow
      - Code examples
   - Keras
   
# Tensorflow Detail



# Problem 1: Use Tensorflow to Implement Deep Learning Network from State-of-The-Art Paper

- CNN


SPP
