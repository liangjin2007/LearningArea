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
   - feed_dict with placeholder
      - a = tf.placeholder(tf.float32, shape=[3])
      - sess.run(b, {a:[1,2,3]})
   - feed_dict with variable
      - a = tf.add(2,5)
      - b = tf.multiply(a,3)
      - sess.run(b, feed_dict={a:15})
   - Operation
      - Constant
         - tf.constant(2)
         - 
      - VariableV2
         - dtype
         - shape
         - create variables
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
         - variable.eval()
         - assign value to variable
         ```
         W = tf.Variable(10)
         W.assign(100)
         with tf.Session() as sess:
             sess.run(W.initializer)
             print(sess.run(W))                    	# >> 10

         W = tf.Variable(10)
         assign_op = W.assign(100)
         with tf.Session() as sess:
             sess.run(assign_op)
             print(W.eval())                     	# >> 100
         ```
         - initializer
      - Idendity
      - Assign
      - Op acts After it is run
      - ![节点符号表](https://github.com/liangjin2007/data_liangjin/blob/master/UnderstandOperationInTheVisualization.jpg?raw=true)
         - This graph contains three separate operations : W/Assign, W, Assign, so need to call sess.run(W.initializer)+sess.run(W) or sess.run(assign_op)+sess.run(W)
   - 
   
   - print graph definition
      - print(tf.get_default_graph().as_graph_def()) 
   - normal loading and lazy loading
      

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
