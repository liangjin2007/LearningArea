# Machine Learning


- PReLU
   - [[2015][iccv]Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852v1.pdf)
   - f(x) = alpha * x for x < 0,  f(x) = x for x>=0
- ELU
   - [[2016][ICLR]FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)](https://arxiv.org/pdf/1511.07289v1.pdf)
   - f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x>=0

- ThresholdedReLU
   - [[2015][ICLR]ZERO-BIAS AUTOENCODERS AND THE BENEFITS OF CO-ADAPTING FEATURES](https://arxiv.org/pdf/1402.3337.pdf)
   - f(x) = x for x > theta,f(x) = 0 otherwise

- Dropout
   - [[2014]Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
   
