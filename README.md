# 18794-final-project
Repository for final project of CMU's 18-794: Pattern Recognition Theory

TO-DO:
Retrain model in KERAS instead of Tensorflow (since knowledge distillation & weight pruning rely on Keras models)
  
  

101-MB Sign Language: https://www.kaggle.com/datamunge/sign-language-mnist

\section{Problem Statement}
American Sign Language is a wholly visual language that is conveyed through hand gestures. Translating this language is a task suited to pattern recognition algorithms and neural networks. However, mobile sign language translator will undoubtedly lack the memory and computational capacity demanded by a desktop-generated neural network. Several techniques exist to shrink these requirements, such as parameter pruning and knowledge distillation, and each has their own advantages and drawbacks. We propose to survey and test a variety of neural network compression techniques[1] in order to evaluate their effectiveness at shrinking a sign language recognition neural network without compromising its accuracy.

\section{Data}
At this time, the dataset has been split into 20592 TRAINING images, 6863 VALIDATION images, and 7172 TEST images.
The data in this project is a dataset of ASL (American Sign Language) letters[2]. This dataset contains 34,627 examples of letters (27,455 training cases and 7172 test cases) but does not contain the letters J or Z (due to these signs having motion in them). There are 24 classes overall (26 total letters minus the two letters with motion gestures) and each of the classes is represented by the number of the alphabet. Each sample of the dataset is a 28x28 grayscale image, which makes it similar to the MNIST database. The data is organized similarly to the MNIST database as well, with the CSV having the labels and all pixel values in a single row for one picture. The dataset was created by taking 1704 color images of the letters, and applying various filters and cropping methods to produce independent results.

\section{Method}
We propose to start by retraining an Inception V3 convolutional neural network on the ASL letter dataset to create a baseline model. Then we will develop three new models: one through performing parameter pruning on the baseline[3][4]; one through performing knowledge distillation on the baseline[5]; one through performing tensor decomposition on the baseline[6]. We propose to rank the speed of the models by measuring the average number of floating-point operations (FLOPS) required by each model to classify one test image.
