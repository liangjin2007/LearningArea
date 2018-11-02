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
