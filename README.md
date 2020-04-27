# deeplearning 역멘토링
2020 ibksystem 플랫폼사업팀 딥러닝 역멘토링 컨텐츠입니다.

## Lecture 1

### 1.1 Machine Learning Introduction

* Machine Learning의 정의
* Machine Learning의 분류
  * 지도학습 (supervised learning)
    * KNN classification
    * linear regression
  * 비지도학습(unsupervised learning)
    * K-means clustering

### 1.2 Linear Regression
* Gradient Descent
  * Learning Rate
* Overfitting
  * Regularization
  * Early stopping
    
### 1.3 Gradient Descent Optimization Algorithms
* Batch Gradient Descent
* Stochastic Gradient Descent (SGD)
* NAG
* Momentum
* Adagrad
* Rmsprop
* Adam

### 1.4 Binary Classification
* Logistic Regression
* Cross-Entropy

### 1.5 Multinomial Classification
* Softmax

### 1.6 실습 코드
* [01.Python Library Tutorial](https://github.com/ibks-hyobin/deeplearning-reverseMentoring/blob/master/Lecture1/01_Python_Library_Tutorial%20(numpy%2Cmatplotlib).ipynb)
* [02.Linear Regression with Python](https://github.com/ibks-hyobin/deeplearning-reverseMentoring/blob/master/Lecture1/02_Linear_Regression.ipynb)
* [03.PyTorch Tutorial](https://github.com/ibks-hyobin/deeplearning-reverseMentoring/blob/master/Lecture1/03_Pytorch_Tutorial.ipynb)
* [04.PyTorch Variable Autograd](https://github.com/ibks-hyobin/deeplearning-reverseMentoring/blob/master/Lecture1/04_variable_autograd.ipynb)
* [05.Linear Regression with PyTorch](https://github.com/ibks-hyobin/deeplearning-reverseMentoring/blob/master/Lecture1/05_Linear_Regression_Models.ipynb)
* [06.Quiz(homework)](https://github.com/ibks-hyobin/deeplearning-reverseMentoring/blob/master/Lecture1/06_Quiz.ipynb)

### 참고 자료
* [stanford university cs231 Lecture 2, Lecture 3](http://cs231n.stanford.edu/2018/syllabus.html)
* [머신러닝, 1시간으로 입문하기](https://www.youtube.com/watch?v=j3za7nv7RfI&t=2047s)
* [머신러닝의 기초 - 선형 회귀 한 번에 제대로 이해하기](https://www.youtube.com/watch?v=ve6gtpZV83E&t=1619s)
* [Gradient Descent Optimization Algorithms 정리](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html)
* [Building a Logistic Regression in Python](https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24)
* [PyTorch로 시작하는 딥러닝 입문](https://wikidocs.net/55580)


## Lecture 2
### 2.1 Introduction to Neural Networks
* History of ANN

### 2.2 Multi-layer Perceptrons (MLP)
* XOR problem
* activation functions
  * gradient vanishing
  * gradient exploding
* Feed-forward Neural Network
* Error Backpropagation

### 2.3 Regularization
* L1 regularization
* L2 regularization
* dropout
* dropconnect
* batch normalization

### 2.4 Weights Initialization
* He initialization
* Xavier Initialization

### 2.5 Hyperparameter tuning

### 2.5 실습 코드
* [01. Multi-Layer Perceptrons](https://github.com/ibks-hyobin/deeplearning-reverseMentoring/blob/master/Lecture1/07_Quiz_Code.ipynb)
* [02. MNIST data classification with MLP](https://github.com/ibks-hyobin/deeplearning-reverseMentoring/blob/master/Lecture2/01.%20pytorch_MNIST_MLP.ipynb)
* [03. Basic MNIST Example](https://github.com/pytorch/examples/tree/master/mnist)

### 참고 자료
* [딥러닝 역사](http://blog.naver.com/PostView.nhn?blogId=windowsub0406&logNo=220883022888)
* [stanford university cs231 Lecture 4](http://aikorea.org/cs231n/optimization-2/)
* [Activation Functions](https://deepestdocs.readthedocs.io/en/latest/002_deep_learning_part_1/0024/)
* [Dropout](https://deepestdocs.readthedocs.io/en/latest/004_deep_learning_part_2/0041/)
* [Batch Normalization](https://sacko.tistory.com/44)
* [Why is logistic regression considered a linear model?](https://www.quora.com/Why-is-logistic-regression-considered-a-linear-model)
* [Why is logistic regression a linear classifier?](https://stats.stackexchange.com/questions/93569/why-is-logistic-regression-a-linear-classifier)

## Lecture 3
### 3.1 Convolution Neural Network
* Convolution
* Channel
* Filter
* Kernel
* Stride
* Padding
* Pooling

### 3.2 CNN Architectures
* LeNet
* **AlexNet**
* **VGGNet**
* GoogLeNet
* **ResNet**
* ZFNet
* DenseNet
* etc

### 3.3 AWS SageMaker
* Using the GPU

### 3.4 실습 코드
* [Pytorch-cifar10 github](https://github.com/kuangliu/pytorch-cifar)

### 참고 자료
* [CNN, Convolutional Neural Network 요약](http://taewan.kim/post/cnn/)
* [라온피플 머신러닝 아카데미 - CNN](https://blog.naver.com/laonple/220587920012?proxyReferer=http%3A%2F%2Fblog.naver.com%2FPostView.nhn%3FblogId%3Dlaonple%26logNo%3D220692793375)

### 4.1 역멘토링 과제 참고 자료
* Pandas
 * [Pandas 공식 홈페이지](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html)
 * [Pandas 10분 완성](https://dataitgirls2.github.io/10minutes2pandas/)
 * [판다스(pandas) 기본 사용법 익히기](https://dandyrilla.github.io/2017-08-12/pandas-10min/)
 
* Matplotlib
 * [Matplotlib 공식 홈페이지](https://matplotlib.org/3.1.1/gallery/index.html)
 * [데이터 사이언스 스쿨 - Pandas의 시각화 기능](https://datascienceschool.net/view-notebook/372443a5d90a46429c6459bba8b4342c/)

* 주택 가격 예측 모델
 * [kaggle - house prices advanced regression techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
 * [kaggle code example(MLP)](https://www.kaggle.com/leostep/pytorch-dense-network-for-house-pricing-regression)
 
* Visulaization example
 * [COVID-19, Analysis, Visualization & Comparisons](https://www.kaggle.com/imdevskp/covid-19-analysis-visualization-comparisons)
 * [Python Data Visualizations](https://www.kaggle.com/benhamner/python-data-visualizations)

## PyTorch Sources
* 참고 자료 : https://github.com/gyunggyung/PyTorch
* Pytorch example : https://github.com/pytorch/examples
* Pytorch tutorial : https://github.com/pytorch/tutorials
* Pytorch documentation : https://pytorch.org/docs/master/
