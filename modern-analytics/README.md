#Applied Machine Learning course

Tracks all resources / assignments / notes etc. for the course 
[Modern Analytics](http://cs5785-cornell-tech.github.io/). 

##Assignments
1. [Assignment0](https://github.com/kulinseth/modern-analytics/tree/master/assignment0): setup and Iris dataset
2. [Assignment1](https://github.com/kulinseth/modern-analytics/tree/master/assignment1):
   1. Digit recognizer: Using kNearest Neighbor (Scikit/own-implementation) and Convnet (Tensorflow). Compare the two
      techniques. Also apply these 2 techniques on CIFAR-10 dataset and Street View House Numbers (SVNH) dataset.
   2. Titanic : kaggle machine learning contest. Logistic regression, SVM etc..
3. [Assignment2](https://github.com/kulinseth/modern-analytics/tree/master/assignment2):
   1. Eigenfaces: SVD/PCA techniques for Face recognition on [Yale Face Dataset](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/Yale%20Face%20Database.htm). 
      Comparing it with CNN visualizations for any match to prinicipal components.
   2. [What's cooking](https://www.kaggle.com/c/whats-cooking): Kaggle competition test out different classification techniques.
      Logistic (discriminative) vs Naive bayes (generative) with Gaussian/Bernoulli priors.
4. [Assignment3](https://github.com/kulinseth/modern-analytics/tree/master/assignment3):
   1. Sentiment analysis: Simple NLP processing pipeline. Algorithm comparison between _Bag of Words_, _2-gram(N-gram)_, _PCA for BoW_ models. Compare these
      models with RNN deep learning model. Dataset used was IMDB_labelled, Yelp_labelled, Amazon_labelled dataset. Applied these
      techniques on Twitter dataset as well.
   2. EM algorithm: Implementation of EM algorithm for Gaussian Mixture Model paramter estimation. Dataset used is [Old
      Failthful Geyser](http://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat). Initialize Gaussian parameters
      using K-means clustering.
5. [Assignment4]():
   1. Association rule learning: Implement the algorithm and apply on [Project Vote Smart](http://api.votesmart.org/docs/index.html) dataset.
   2. Random forests: Implement the algorithm and apply it for Image approximation.

##Written Exercises
Written exercises were part of all assignments comprising of 3 questions. They are written-up in Tex format and attached as PDFs
along with the .tex files. Questions covered different topics like 
1. Linear algebra
   1. Eigenvalue problem, SVD of rank deficient matrix
   2. LDA and least-squares correspondence 
2. Basic probability and stats
   1. Gradient and Hessian of log-likelihood of Logistic regression
   2. Application of Bayes rule
3. Unsup
   1. GMM and EM algorithm details
   2. Procrustes algorithm
   3. Multidimensional scaling.
4. Application
   1. Association rule learning
   2. Neural network as Function approx.
   3. [ConvnetJS](http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html) 

##Course Final/Project
[Scene classification](https://inclass.kaggle.com/c/cornell-tech-cs5785-2015-fall-final) with following Data provided
1. Supervised dataset:
    1. Alexnet CNN codes for the images
    2. SIFT image feature vectors for Bag of Visual words model
    3. Attribute data about the images, in the form of binary attribute vectors that indicate
       the presence or absence of certain key aspects of the image ("symmetrical," "open area", "horizon", etc)
2. Unsupervised dataset:
    1. 10K similar images with 5 captions.
Finally a report is written in NIPS paper template describing the details of approaches and results achieved.

[Approaches evaluated](https://github.com/kulinseth/modern-analytics/tree/master/final) to solve the problem:
1. Simple supervised techniques:
    1. Softmax on the Alexnet CNN feature vectors
    2. K-means on SIFT descriptors to get the visual dicitonary. Histogram of these visual words can be used 
    in [Bag of Words](http://www.computervisionblog.com/2015/01/from-feature-descriptors-to-deep.html) model. Another idea is to use Pyramid Match kernel scheme.
    3. Train your own CNN feature vector using VGG model.
    4. [Transfer learning](https://www.tensorflow.org/versions/r0.7/tutorials/image_recognition/index.html)
2. Semi-supervised learning techniques
    1. Use the 10K image dataset with captions to train the RNN. Use the trained model to generate captions for the Training
       and Test images. Using these captions generate a Bag of Words dictionary and generate a model which can be used to classify the images.
    2. Use the Attribute data to train the CNN feature vectors by using that information to provide metadata. This can be used
       to generate more training data.
    3. Use 10K SIFT features 

##Books used
Required: 
T. Hastie, R. Tibshirani and J. Friedman, The Elements of Statistical Learning: Data Mining, Inference, and Prediction (2nd edition), Springer-Verlag, 2008.

Recommended: 
P. Harrington, Machine Learning in Action, Manning, 2012.
A. Rajaraman, J. Leskovec and J. Ullman, [Mining of Massive Datasets](http://web.stanford.edu/class/cs246/handouts.html), v1.1.
H. Daumé III, A Course in Machine Learning, v0.8.



