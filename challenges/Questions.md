# ML interview questions:

1. What is A/B testing, explain using an example?
   Why do we need hypothesis testing? What is P-Value?
   What is the null hypothesis? How do we state it?
   Do you know what Type-I/Type-II errors are?

   The A/B testing is a randomized experiment done to test 2 samples namely A
   and B of a random variable. Its a form of statistics hypothesis testing,
   where A is a control and B is experiment to test the response of a subject.
   The response is measured using some Statistic.
   Example could be the change in the design of a website or UI element which
   results in improvement of customer drop-off rates.

   Statistic used to measure:
   * Welch's t-test, this assumes the least. It allows for unequal variances,
     sample sizes etc.

     t = (X1_mean - X2_mean)/ sqrt( s1^2/N1 + s2^2/N2)

     si^2 is the variance of the sample sizes
     Ni is the size
   * Student t-test : one sample and two sample t-tests. One sample is when
     single population mean is compared with the null-hypothesis. And two sample
     is when two population means are compared.

     one sample t = (X_mean - mu (null hypothesis mean)) / (s/sqrt(n))
     s = standard deviation

     two sample t = (X1_mean - X2_mean) / sqrt(s1^2 + s2^2 / 2)

   * Z-test: here both the means and variances are expected to be same

   Hypothesis testing: basically means certain hypothesis is testable on the
   basis of observing certain process which is modeled using random variables
   H => F (z, Theta)

   The hypothesis metrics which estimate the significance or lack there of, is
   one way to evaluate the hypothesis testing. The other way is to come up with
   different statistical models for each candidate hypothesis and use model
   selection to use the best model. AIC or Bayes factor can be used to do model
   selection and testing.

   The two datasets are compared or sampled dataset is compared with a synthetic
   dataset which is generated using modelling or math in an idealized way.

   *Null hypothesis* : when there is no relationship between 2 datasets
   *Type I error*: when null-hypothesis is wrongly rejected. Alternate theory was
   considered, even though it was not significant.

   *Type II error*: when the null-hypothesis is wrongly not rejected.


2. An important metric goes down, how would you dig into the causes?

   Depending on metric, we would start looking at what changed? For instance in
   case of MSE : Sigma(y - y\_hat)^2 / N

   Here the ground truth y is same, but y\_hat changed, that leads to some
   parameter values got updated. Was it a new model ? or there were new set of
   sample values which were not great. Look at the features which most affect
   that metric by getting confidence interval or determining the importance of
   that feature in the result. Maybe new features got added or removed which
   skewed the balance.

3. How do you remove the missing values from a data set. What if it causes bias? What will you do then?

   *TODO*
   I think its asking about imputation, how to fill the missing values. Or maybe
   drop those samples altogether. If it leads to class imbalance, how to remove
   bias ?

4. Design a metric that help reduce bias in the data set.

   Individual fairness vs group fairness: Here it means that different groups
   should be treated equally or fairly, as compared to when individuals who are
   similar should be treated equally. its hard to achieve both, as in case of
   affirmative action.

   Balanced vs Imbalanced fairness:  ground truths between different groups is
   balanced and representative. Like Apple was in the news for the imbalance in
   face recognition between while male vs black female

   Sample bias vs Label bias: If the groups are getting assigned different
   labels for same behavior , same white vs black
   sample bias occurs when data generation process samples from different groups
   differently.

   Measures to alleviate the issue:
   1. Look at whether the ground truths are balanced or not across different
      groups.
   2. Think about the process which generated the data
   3. Keep humans in the loop

   4. Have more diverse selection of people in the group.
   5. periodically test the dataset that its representative.

5. How would you impute missing information in a dataset?
   *TODO*
6. Find the potential causes of an anomaly in web traffic dataset.

   Does it coincide with some other timed event.

7. Explain Logistic regression.
8. Company research ...
9. What metrics will you evaluate based on a scenario? — (e.g. Launch in new city)
10. How can you report the statistical intensive results to a non-statistician group?
11. Explain what regularization is and why it is useful.

   Regularization helps reduce overfitting in the model. When you have
   over-parameterized model as compared to the dimentionality of the data, it
   can result in highly complex models. This helps reduce the complexity. The
   way it does that is using L1, L2, Lp norm of the weights in the loss
   function. You specify them and the loss function penalizes it.

   Regularization helps reduce the variance of the model without substantially
   increading the variance of the problem.

   Y = b0 + b1X1 + b2X2 + b3X3 ...

   Residual Sum of Squares RSS = (Y - b0 - Sig(biXi))^2
   Ridge regression = RSS + Sig(b^2)
   Lasso regression = RSS + L1Norm
   [Source](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)

12. If you could take advantage of multiple CPU cores, would you prefer
    a boosted-tree algorithm over a random forest? Why?
    (Hint: if you have 10 hands to do a task, you take advantage of it)

    Both are ensemble methods.

    Gradient Boosted tree algorithm: this seems like an iterative algorithm.
    Here the trees are built taking into account the errors made by previously
    trained tree.

Background questions:

1. Summarize your experience

   Systems engineer with interest in Machine learning.

2. What companies you worked at? What was your role?
3. Do you have a project portfolio? What projects you implemented? Discuss some of them in details
   Here add the Github projects on the website and highlight the important ones.

   Projects :

   a. Transfer learning in the context of medical datasets. The idea 
   b. Network introspection one using perturbation techniques
   c. SegTHOR, learn a new loss function.

   In the SegTHOR project, we worked on developing a new loss function which
   takes into account the shape based information. The project is about the
   multi-organ segmentation in abdominal area. The main issue was that the
   neighboring surrounding organs were getting misclassified and the individual
   organ metrics were not looking good. The process was we started with a basic
   U-net multiclass segementation model. There are many ways to design the loss
   function here, in the form of structured loss. It could be topological
   constraints, graph-based constraints, energy based approaches. We started
   with cross-entropy loss, generalized dice loss, DICE hinge loss...

   Cross-entropy :

   Metrics: DICE, jaccard, hausdorff-distance..
      DICE:
      jaccard: like hamming ?

   d. cleanup the Openslide and talk about the Camelyon challenge
   e. the ML course projects and what I implemented in them

4. For aspiring data scientists: Why do you want a career in data science?

   I enjoy working on it. The ML techniques if applied correctly have a lot of
   potential to supplant currently available technologies and make a difference,
   that excites me to contribute to it. Besides there are lot of challenging
   open problems still exsist.

5. Have you taken any data-science-related online courses? If yes, how many did you complete with a certificate?
   Yes. MIT one, NEU one, Coursera one.
6. Have you participated in any data science challenges? If yes, can you describe one of them?
   Can talk about the SegTHOR one. We missed it but we have quality work out of
   it.

Process
All Machine Learning, Data Mining and Data Science projects should follow some process, so there can be questions about it:

1. Can you outline the steps in a data science project?

   Problem Definition : what's the objective of the problem we are trying to solve
   Data Understanding (or Data Exploration): Visualize few parameters using
                  Notebooks, Do some histogram analysis. Check the data and label
                  distribution. Study where the data is coming from?
   Data Preparation :  Like for instance in CT and pathology look at the scanner
                     details. This sheds light on if there is some skew
   Modeling : start with a simple model to come up with a baseline, like Logistic
         regression and then go on to apply more fancy techniques..
   Evaluation : need metrics
   Deployment (for the production) : doable...

   What is the goal of each step?
   What are possible activities at each step?

Mathematics
Linear Algebra

1. What is Ax = b ? How to solve it?

   TODO: There are two ways to think about it:
   a. One is that its linear combination of Rm vectors, where A is m x n, x is
      Rn and b is Rm
   b. The other is that it projects the n dimensional vectors to the m one -
   check this
2. How do we multiply matrices?

   Strassen matrix multiplication :
      It works by recursively partitioning the matrices and applying the product
      and addition in the submcomponents.
3. What is an Eigenvalue? And what is an Eigenvector? What is Eigenvalue Decomposition or The Spectral Theorem?
4. What is Singular Value Decomposition?


## Statistics
Probability

1. Given two fair dices, what is the probability of getting scores that sum to 4? to 8?
2. A simple questions on Bayes rule:
   Imagine a test with a true positive rate of 100% and false positive rate of 5%.
   Imagine a population with a 1/1000 rate of having the condition the test identifies.
   Given a positive test, what is the probability of having that condition?

Distributions
You may expect questions about probability distributions:

* What is the normal distribution? Give an example of some variable that follows this distribution
* What about log-normal?
* Explain what a long tailed distribution is and provide three examples of relevant phenomena that have long tails. Why are they important in classification and prediction problems?
* How to check if a distribution is close to Normal? Why would you want to check it? What is a QQ Plot?
* Give examples of data that does not have a Gaussian distribution, or log-normal.
* Do you know what the exponential family is?
* Do you know the Dirichlet distribution? the multinomial distribution?

Basic Statistics
* What is the Laws of Large Numbers? Central Limit Theorem?
* Why are they important for Statistics?
* What summary statistics do you know?

Experiment Design
* Designing experiments is an important part of Statistics, and it’s especially useful for doing A/B tests.

###Sampling and Randomization

Why do we need to sample and how?
Why is randomization important in experimental design?
Some 3rd party organization randomly assigned people to control and experiment groups. How can you verify that the assignment truly was random?
How do you calculate needed sample size?
Power analysis. What is it?
Biases

When you sample, what bias are you inflicting?
How do you control for biases?
What are some of the first things that come to mind when I do X in terms of biasing your data?
Other questions

What are confounding variables?
Point Estimates
Confidence intervals

What is a point estimate? What is a confidence interval for it?
How are they constructed?
How to interpret confidence intervals?
Testing
Hypothesis tests

What is -Test/-Test/ANOVA? When to use it?
How would you test if two populations have the same mean? What if you have 3 or 4 populations?
You applied ANOVA and it says that the means are different. How do you identify the populations where the differences are significant?
What is the distribution of p-value’s, in general?
A/B Tests
What is A/B testing? How is it different from usual Hypothesis testing?
How can you prove that one improvement you’ve brought to an algorithm is really an improvement over not doing anything? How familiar are you with A/B testing?
How can we tell whether our website is improving?
What are the metrics to evaluate a website? A search engine?
What kind of metrics would you track for you music streaming website?
Common metrics: Engagement / retention rate, conversion, similar products / duplicates matching, how to measure them.
Real-life numbers and intuition: Expected user behavior, reasonable ranges for user signup / retention rate, session length / count, registered / unregistered users, deep / top-level engagement, spam rate, complaint rate, ads efficiency.
Time Series
What is a time series?
Did you do any projects which involved dealing with time?
What is the difference between data for usual statistical analysis and time series data?
Have you used any of the following: Time series models, Cross-correlations with time lags, Correlograms, Spectral analysis, Signal processing and filtering techniques? If yes, in which context?
In time series modeling how can we deal with multiple types of seasonality like weekly and yearly seasonality?
Advanced
Resampling

Explain what resampling methods are. Why they are useful. What are their limitations?
Bootstrapping - how and why it is used?
How to use resampling for hypothesis testing? Have you heard of Permutation Tests?
How would you apply resampling to time series data?

##Machine Learning

General ML Questions
The ML part may start with something like:

What is the difference between supervised and unsupervised learning? Which algorithms are supervised learning and which are not? Why?
What is your favorite ML algorithm and why?
And then go into details

Regression
Describe the regression problem. Is it supervised learning? Why?
What is linear regression? Why is it called linear?
Discuss the bias-variance tradeoff.
Linear Regression:

What is Ordinary Least Squares Regression? How it can be learned?
Can you derive the OLS Regression formula? (For one-step solution)
Is model still linear? Why?
Do we always need the intercept term? When do we need it and when do we not?
What is collinearity and what to do with it? How to remove multicollinearity?
What if the design matrix is not full rank?
What is overfitting a regression model? What are ways to avoid it?
What is Ridge Regression? How is it different from OLS Regression? Why do we need it?
What is Lasso regression? How is it different from OLS and Ridge?
Linear Regression assumptions:

Function is a linear combination of the x dimension values. 
RSS: residual sum of squares..
What are the assumptions required for linear regression?
What if some of these assumptions are violated?
Significant features in Regression

You would like to find significant features. How would you do that?
You fit a multiple regression to examine the effect of a particular feature. The feature comes back insignificant, but you believe it is significant. Why can it happen?
Your model considers the feature significant, and is not, but you expected the opposite result. Why can it happen?
Evaluation

How to check is the regression model fits the data well?
Other algorithms for regression

Decision trees for regression
-Nearest Neighbors for regression. When to use?
Do you know others? E.g. Splines? LOESS/LOWESS?
Classification
Basic:

Can you describe what is the classification problem?
What is the simplest classification algorithm?
What classification algorithms do you know? Which one you like the most?
Decision trees:

What is a decision tree?
What are some business reasons you might want to use a decision tree model?
How do you build it? What impurity measures do you know?
Describe some of the different splitting rules used by different decision tree algorithms.
Is a big brushy tree always good? Why would you want to prune it?
Is it a good idea to combine multiple trees?
What is Random Forest? Why is it good?
Other ways to combine trees? What about boosting?
Logistic regression:

What is logistic regression?
How do we train a logistic regression model?
How do we interpret its coefficients?
Support Vector Machines

What is the maximal margin classifier? How this margin can be achieved and why is it beneficial?
How do we train SVM? What about hard SVM and soft SVM?
What is a kernel? What's the intuition behind the Kernel trick?
Which kernels do you know? How to choose a kernel?
Neural Networks

What is an Artificial Neural Network?
How to train an ANN? What is back propagation?
How does a neural network with three layers (one input layer, one inner layer and one output layer) compare to a logistic regression?
What is deep learning? What is CNN (Convolution Neural Network) or RNN (Recurrent Neural Network)?
Other models:

What other models do you know?
How can we use Naive Bayes classifier for categorical features? What if some features are numerical?
Tradeoffs between different types of classification models. How to choose the best one?
Compare logistic regression with decision trees and neural networks.
Regularization
What is Regularization?
Which problem does Regularization try to solve?
What does it mean (practically) for a design matrix to be “ill-conditioned”?
When might you want to use ridge regression instead of traditional linear regression?
What is the difference between the and regularization?
Why (geometrically) does LASSO produce solutions with zero-valued coefficients (as opposed to ridge)?
Let us go through the derivation of OLS or Logistic Regression. What happens when we add regularization? How do the derivations change? What if we replace regularization with regularization?
Dimensionality Reduction
Basics:

What is the purpose of dimensionality reduction and why do we need it?
Are dimensionality reduction techniques supervised or not? Are all of them are (un)supervised?
What ways of reducing dimensionality do you know?
Is feature selection a dimensionality reduction technique?
What is the difference between feature selection and feature extraction?
Is it beneficial to perform dimensionality reduction before fitting an SVM? Why or why not?
Principal Component Analysis:

What is Principal Component Analysis (PCA)? What is the problem it solves? How is it related to eigenvalue decomposition (EVD)?
What’s the relationship between PCA and SVD? When SVD is better than EVD for PCA?
Under what conditions is PCA effective?
Why do we need to center data for PCA and what can happed if we don’t do it? Do we need to scale data for PCA?
Is PCA a linear model or not? Why?
Other Dimensionality Reduction techniques:

Do you know other Dimensionality Reduction techniques?
What is Independent Component Analysis (ICA)? What’s the difference between ICA and PCA?
Suppose you have a very sparse matrix where rows are highly dimensional. You project these rows on a random vector of relatively small dimensionality. Is it a valid dimensionality reduction technique or not?
Have you heard of Kernel PCA or other non-linear dimensionality reduction techniques? What about LLE (Locally Linear Embedding) or -SNE (-distributed Stochastic Neighbor Embedding)
What is Fisher Discriminant Analysis? How it is different from PCA? Is it supervised or not?
Cluster Analysis
What is the cluster analysis problem?
Which cluster analysis methods you know?
Describe -Means. What is the objective of -Means? Can you describe the Lloyd algorithm?
How do you select for K-Means?
How can you modify -Means to produce soft class assignments?
How to assess the quality of clustering?
Describe any other cluster analysis method. E.g. DBSCAN.
Optimization
You may have some basic questions about optimization:

What is the difference between a convex function and non-convex?
What is Gradient Descent Method?
Will Gradient Descent methods always converge to the same point?
What is a local optimum?
Is it always bad to have local optima?
Recommendation
What is a recommendation engine? How does it work?
Do you know about the Netflix Prize problem? How would you approach it?
How to do customer recommendation?
What is Collaborative Filtering?
How would you generate related searches for a search engine?
How would you suggest followers on Twitter?
Feature Engineering
How to apply Machine Learning to audio data, images, texts, graphs, etc?
What is Feature Engineering? Can you give an example? Why do we need it?
How to go from categorical variables to numerical?
What to do with categorical variables of high cardinality?
Natural Language Processing
If the company deals with text data, you can expect some questions on NLP and Information Retrieval:

What is NLP? How is it related to Machine Learning?
How would you turn unstructured text data into structured data usable for ML models?
What is the Vector Space Model?
What is TF-IDF?
Which distances and similarity measures can we use to compare documents? What is cosine similarity?
Why do we remove stop words? When do we not remove them?
Language Models. What is -Grams?
What is word2vec? How it can be used in NLP and IR?
Meta Learning
Feature Selection:

Are all features equally good?
What are the downfalls of using too many or too few variables?
How many features should you use? How do you select the best features?
What is Feature Selection and why do we need it?
Describe several feature selection methods. Are these methods depend on the model or not?
Model selection:

You have built several different models. How would you select the best one?
You have one model and want to find the best set of parameters for this model. How would you do that?
How would you look for the best parameters? Do you know something else apart from grid search?
What is Cross-Validation?
What is 10-Fold CV?
What is the difference between holding out a validation set and doing 10-Fold CV?
Model evaluation

How do you know if your model overfits?
How do you assess the results of a logistic regression?
Which evaluation metrics you know? Something apart from accuracy?
Which is better: Too many false positives or too many false negatives?
What precision and recall are?
What is a ROC curve? What is AU ROC (AUC)? How to interpret the curve and AU ROC?
Do you know about Concordance or Lift?
Discussion Questions:

You have a marketing campaign and you want to send emails to users. You developed a model for predicting if a user will reply or not. How can you evaluate this model? Is there a chart you can use?
Miscellanea
Curse of Dimensionality

What is Curse of Dimensionality? How does it affect distance and similarity measures?
What are the problems of large feature space? How does it affect different models, e.g. OLS? What about computational complexity?
What dimensionality reductions can be used for preprocessing the data?
What is the difference between density-sparse data and dimensionally-sparse data?
Others

You are training an image classifier with limited data. What are some ways you can augment your dataset?
Computer Science
Knowledge in Computer Science is as important for Data Science as knowledge in Machine Learning. So you may get the same type of questions as for any software developer position, but possibly with lower expectations on your answers.

I was a Java developer for quite some time, and I prepared a list of questions I asked (and often was asked) on Java interviews: Java Inteview questions. This list can also be helpful for preparing to a Data Science interview.

Libraries and Tools
Apart from basics of Java/Scala/Python/etc, you may be asked about libraries for data analysis:

Which libraries for data analysis do you know in Python/R/Java?
Have you used numpy, scipy, pandas, sklearn?
What are some features of the sklearn api that differentiate it from fitting models in R?
What are some features of pandas/sklearn that you like? Don't like? Same questions for R.
Why is “vectorization” such a powerful method for optimizing numerical code? What is going on that makes the code faster relative to alternatives like nested for loops?
When is it better to write your own code than using a data science software package?
State any 3 positive and negative aspects about your favorite statistical software.
Describe a difficult bug you’ve encountered and how you resolved it.
How does floating point affect precision of calculations? Equality tests?
What is BLAS? LAPACK?
Databases
Have you been involved in database design and data modeling?
SQL-Related questions: e.g. what is "group by"?
Or given some DB schema you may be asked to write a simple SQL query.
What is a “star schema”? “snowflake schema”?
Describe different NoSQL technologies you’re familiar with, what they are good at, and what they are bad at.
Distributed Systems and Big Data
Basic “Big Data” questions:

What is the biggest data set that you have processed and how did you process it? What was the result?
Have you used Apache Hadoop, Apache Spark, Apache Flink? Why? Have you used Apache Mahout?
MapReduce

What is MapReduce? Why is it “shared-nothing” architecture?
Can you implement word count in MapReduce? What about something a bit more complex like TF-IDF? Naive Bayes?
What is load balance? How to make sure a MapReduce application has good load balance?
Can you give examples where MapReduce does not work?
What are examples of “embarassingly parallelizable” algorithms?
How would you estimate the median of a dataset that is too big to hold in the memory?
Implementation questions

There are some posts that you may find useful when preparing for the “Big Data” part:

Hands-On
Also, many interviews have a part which I call “hands-on”: you are given some problem description and you are asked to solve it. You can just talk the interviewers through your solution or even be asked to sit and implement some parts. Sometimes there is also a test assignment to be done at home (prior to the interview).

Problem to Solve
For example:

Assume that you are asked to lead a project on churn detection, and have dataset of known users who stopped using the service and ones who are still using. This data includes demographics and other features.

Do the following:

Describe the methodology and model that you will chose to identify churn, and describe your thought process.
Think how would you communicate the results to the CEO?
Suppose in the dataset only 0.025 of users churned. How would you make it more balanced?
Also:

How would you implement it if you had one day? One month? One year?
How would your approach scale?
Other problems:

How would you approach identifying plagiarism?
How to find individual paid accounts shared by multiple users?
How to detect bogus reviews, or bogus Facebook accounts used for bad purposes?
Usually the domain of the problem is related to what the company is doing. If they’re doing marketing, it will most likely be marketing related.
Additionally, you may be asked:

How would you approach collecting the data if you didn’t have the dataset?
Coding
Sometimes you even may be presented a small dataset and ask to do a particular task with any tool. For example,

write a script to extract features,
then do some exploratory data analysis and
finally apply some ML algorithm to this dataset.
Or just the last two, with a ready to use dataset in tabular form.

Sources
I had to work through a lot of sources to make this compilation. I did not include all the questions I came across,
just the ones that made sense or ones I really got during my interviews. It also, of course, includes my own interviews.

Facebook Data Science Interview Questions
1)         A building has 100 floors. Given 2 identical eggs, how can you use them to find the threshold floor? The egg will break from any particular floor above floor N, including floor N itself.

2)         In a given day, how many birthday posts occur on Facebook?

3)         You are at a Casino. You have two dices to play with. You win $10 every time you roll a 5. If you play till you win and then stop, what is the expected pay-out?

4)         How many big Macs does McDonald sell every year in US?

5)         You are about to get on a plane to Seattle, you want to know whether you have to bring an umbrella or not. You call three of your random friends and as each one of them if it’s raining. The probability that your friend is telling the truth is 2/3 and the probability that they are playing a prank on you by lying is 1/3. If all 3 of them tell that it is raining, then what is the probability that it is actually raining in Seattle.

6)         You can roll a dice three times. You will be given $X where X is the highest roll you get. You can choose to stop rolling at any time (example, if you roll a 6 on the first roll, you can stop). What is your expected pay-out?

7)         How can bogus Facebook accounts be detected?

8)       You have been given the data on Facebook user’s friending or defriending each other. How will you determine whether a given pair of Facebook users are friends or not?

9)         How many dentists are there in US?

10)         You have 2 dices. What is the probability of getting at least one 4? Also find out the probability of getting at least one 4 if you have n dices.

11)       Pick up a coin C1 given C1+C2 with probability of trials p (h1) =.7, p (h2) =.6 and doing 10 trials. And what is the probability that the given coin you picked is C1 given you have 7 heads and 3 tails? 

12)     You are given two tables- friend_request and request_accepted. Friend_request contains requester_id, time and sent_to_id and request_accepted table contains time, acceptor_id and requestor_id. How will you determine the overall acceptance rate of requests?

13)       How would add new Facebook members to the database of members, and code their relationships to others in the database? 

14)       What would you add to Facebook and how would you pitch it and measure its success?

15)  How will you test that there is increased probability of a user to stay active after 6 months given that a user has more friends now?

16) You have two tables-the first table has data about the users and their friends, the second table has data about the users and the pages they have liked. Write an SQL query to make recommendations using pages that your friends liked. The query result should not recommend the pages that have already been liked by a user.

17) What is the probability of pulling a different shape or a different colour card from a deck of 52 cards?

18) Which technique will you use to compare the performance of two back-end engines that generate automatic friend recommendations on Facebook?

19) Implement a sorting algorithm for a numerical dataset in Python.

20) How many people are using Facebook in California at 1.30 PM on Monday?

21) You are given 50 cards with five different colors- 10 Green cards, 10 Red Cards, 10 Orange Cards, 10 Blue cards, and 10 Yellow cards. The cards of each colors are numbered from one to ten. Two cards are picked at random. Find out the probability that the cards picked are not of same number and same color.

22) What approach will you follow to develop the love,like, sad feature on Facebook?

Insight Data Science Interview Questions
1)         Which companies participating in Insight would you be interested in working for? 

2)         Create a program in a language of your choice to read a text file with various tweets. The output should be 2 text files-one that contains the list of all unique words among all tweets along with the count for repeated words and the second file should contain the medium number of unique words for all tweets.

3)         What motivates you to transition from academia to data science?


Twitter Data Scientist Interview Questions                       
1)    How can you measure engagement with given Twitter data?

2)    Give a large dataset, find the median.

3)    What is the good measure of influence of a Twitter user?

AirBnB Data Science Interview Questions
1)  Do you have some knowledge of R - analyse a given dataset in R?

2)  What will you do if removing missing values from a dataset cause bias?

3)  How can you reduce bias in a given data set?

4) How will you impute missing information in a dataset?

Google Data Science Interview Questions
1)  Explain about string parsing in R language

2) A disc is spinning on a spindle and you don’t know the direction in which way the disc is spinning. You are provided with a set of pins.How will you use the pins to describe in which way the disc is spinning?

3)  Describe the data analysis process.

4) How will you cut a circular cake into 8 equal pieces?

LinkedIn Data Science Interview Questions
1)  Find out K most frequent numbers from a given stream of numbers on the fly.

2)  Given 2 vectors, how will you generate a sorted vector?

3)  Implementing pow function

4)  What kind of product you want to build at LinkedIn?

5)  How will you design a recommendation engine for jobs?

6)  Write a program to segment a long string into a group of valid words using Dictionary. The result should return false if the string cannot be segmented. Also explain about the complexity of the devised solution.

7) Define an algorithm to discover when a person is starting to search for new job.

8) What are the factors used to produce “People You May Know” data product on LinkedIn?

9)  How will you find the second largest element in a Binary Search tree ? (Asked for a Data Scientist Intern job role)

 

Master Machine Learning with interesting machine learning project ideas

 
Mu Sigma Data Science Interview Questions
1)   Explain the difference between Supervised and Unsupervised Learning through examples.

2)   How would you add value to the company through your projects?

3)   Case Study based questions – Cars are implanted with speed tracker so that the insurance companies can track about our driving state. Based on this new scheme what kind of business questions can be answered?

4)  Define standard deviation, mean, mode and median.

5) What is a joke that people say about you and how would you rate the joke on a scale of 1 to 10?

6) You own a clothing enterprise and want to improve your place in the market. How will you do it from the ground level ?

7) How will you customize the menu for Cafe Coffee Day ?



Amazon Data Science Interview Questions
1) Estimate the probability of a disease in a particular city given that the probability of the disease on a national level is low.

2) How will inspect missing data and when are they important for your analysis?

3) How will you decide whether a customer will buy a product today or not given the income of the customer, location where the customer lives, profession and gender? Define a machine learning algorithm for this.

4) From a long sorted list and a short 4 element sorted list, which algorithm will you use to search the long sorted list for 4 elements.

5) How can you compare a neural network that has one layer, one input and output to a logistic regression model?

6) How do you treat colinearity?

7) How will you deal with unbalanced data where the ratio of negative and positive is huge?

8) What is the difference between -

i) Stack and Queue

ii) Linkedin and Array

Uber Data Science Interview Questions
1) Will Uber cause city congestion?

2) What are the metrics you will use to track if Uber’s paid advertising strategies to acquire customers work? How will you figure out the acceptable cost of customer acquisition?

3) Explain principal components analysis with equations.

4) Explain about the various time series forecasting technqiues.

5) Which machine learning algorithm will you use to solve a Uber driver accepting  request?

6)How will you compare the results of various machine learning algorithms?

7) How to solve multi-collinearity?

8) How will you design the heatmap for Uber drivers to provide recommendation on where to wait for passengers? How would you approach this?

9) If we added one rider to the current SF market, how would that affect the existing riders and drivers?  

10) What are the different performance metrics for evaluating Uber services?

11) How will you decide which version (Version 1 or Version 2) of the Surge Pricing Algorithms is working better for Uber ?

12) How will you explain JOIN function in SQL to a 10 year old ?

Netflix Data Science Interview Questions
1) How can you build and test a metric to compare ranked list of TV shows or Movies for two Netflix users?

2) How can you decide if one algorithm is better than the other?

Microsoft Data Science Interview Questions
1) Write a function to check whether a particular word is a palindrome or not.

2) How can you compute an inverse matrix faster by playing with some computation tricks?

3) You have a bag with 6 marbles. One marble is white.  You reach the bag 100 times. After taking out a marble, it is placed back in the bag. What is the probability of drawing a white marble at least once?

Apple Data Science Interview Questions
1) How do you take millions of users with 100's of transactions each, amongst 10000's of products and group the users together in a meaningful segments?

Adobe Data Scientist Interview Questions
1) Check whether a given integer is a palindrome or not without converting it to a string.

2) What is the degree of freedom for lasso?

3) You have two sorted array of integers, write a program to find a number from each array such that the sum of the two numbers is closest to an integer i.

American Express Data Scientist Interview Questions
1) Suppose that American Express has 1 million card members along with their transaction details. They also have 10,000 restaurants and 1000 food coupons. Suggest a method which can be used to pass the food coupons to users given that some users have already received the food coupons so far.

2) You are given a training dataset of users that contain their demographic details, the pages on Facebook they have liked so far and results of psychology test  based on their personality i.e. their openness to like FB pages or not. How will you predict the age, gender and other demographics of unseen data?

Quora Data Scientist Interview Questions
1) How will you test a machine learning model for accuracy?

2) Print the elements of a matrix in zig-zag manner.

3) How will you overcome overfitting in predictive models?

4) Develop an algorithm to sort two lists of sorted integers into a single list.

Goldman Sachs Data Scientist Interview Questions
1) Count the total number of trees in United States.

2) Estimate the number of square feet pizza’s eaten in US each year.

3) A box has 12 red cards and 12 black cards. Another box has 24 red cards and 24 black cards. You want to draw two cards at random from one of the two boxes, which box has a higher probability of getting cards of same colour and why?

4) How will you prove that the square root of 2 is irrational?

5) What is the probability of getting a HTT combination before getting a TTH combination?

6) There are 8 identical balls and only one of the ball is slightly heavier than the others. You are given a balance scale to find the heavier ball. What is the least number of times you have to use the balance scale to find the heavier ball?

Walmart Data Science Interview Questions
1) Write the code to reverse a Linked list.

2) What assumptions does linear regression machine learning algorithm make?

3) A stranger uses a search engine to find something and you do not know anything about the person. How will you design an algorithm to determine what the stranger is looking for just after he/she types few characters in the search box?

4) How will you fix multi-colinearity in a regression model?

5) What data structures are available in the Pandas package in Python programming language?

6) State some use cases where Hadoop MapReduce works well and where it does not.

7) What is the difference between an iterator, generator and list comprehension in Python?

8) What is the difference between a bagged model and a boosted model?

9) What do you understand by parametric and non-parametric methods? Explain with examples.

10) Have you used sampling? What are the various types of sampling have you worked with?

11) Explain about cross entropy ?

12) What are the assuptions you make for linear regression ?

13) Differentiate between gradient boosting and random forest.

14) What is the signigicance of log odds ?

IBM Data Science Interview Questions
1) How will you handle missing data ?

Yammer Data Science Interview Questions
How can you solve a problem that has no solution?
On rolling a dice if you get $1 per dot on the upturned face,what are your expected earnings from rolling a dice?
In continuation with question #2, if you have 2 chances to roll the dice and you are given the opportunity to decide when to stop rolling the dice (in the first roll or in the second roll). What will be your rolling strategy to get maximum earnings?
 What will be your expected earnings with the two roll strategy?
You are creating a report for user content uploads every month and observe a sudden increase in the number of upload for the month of November. The increase in uploads is particularly in image uploads. What do you think will be the cause for this and how will you test this sudden spike?
Citi Bank Data Science Interview Questions
1) A dice is rolled twice, what is the probability that on the second chance it will be a 6?

2) What are Type 1 and Type 2 errors ?

3) Burn two ropes, one needs 60 minutes of time to burn and the other needs 30 minutes of time. How will you achieve this in 45 minutes of time ?

Data Science Interview Questions Asked at Other Top Tech Companies
1) R programming language cannot handle large amounts of data. What are the other ways of handling it without using Hadoop infrastructure? (Asked at Pyro Networks)

2) Explain the working of a Random Forest Machine Learning Algorithm (Asked at Cyient)

3) Describe K-Means Clustering.(Asked at Symphony Teleca)

4) What is the difference between logistic and linear regression? (Asked at Symphony Teleca)

5) What kind of distribution does logistic regression follow? (Asked at Symphony Teleca)

6) How do you parallelize machine learning algorithms? (Asked at Vodafone)

7) When required data is not available for analysis, how do you go about collecting it? (Asked at Vodafone)

8) What do you understand by heteroscadisticity (Asked at Vodafone)

9) What do you understand by confidence interval? (Asked at Vodafone)

10) Difference between adjusted r and r square. (Asked at Vodafone)

11) How Facebook recommends items to newsfeed? (Asked at Finomena)

12)  What do you understand by ROC curve and how is it used? (Asked at MachinePulse)

13) How will you identify the top K queries from a file? (Asked at BloomReach)

14) Given a set of webpages and changes on the website, how will you test the new website feature to determine if the change works positively? (Asked at BloomReach)

15) There are N pieces of rope in a bucket. You put your hand into the bucket, take one end piece of the rope .Again you put your hand into the bucket and take another end piece of a rope. You tie both the end pieces together. What is the expected value of the number of loops within the bucket? (Asked at Natera)

16) How will you test if a chosen credit scoring model works or not? What data will you look at? (Asked at Square)

17) There are 10 bottles where each contains coins of 1 gram each. There is one bottle of that contains 1.1 gram coins. How will you identify that bottle after only one measurement? (Data Science Puzzle asked at Latent View Analytics)

18) How will you measure a cylindrical glass filled with water whether it is exactly half filled or not? You cannot measure the water, you cannot measure the height of the glass nor can you dip anything into the glass. (Data Science Puzzle asked at Latent View Analytics)

19) What would you do if you were a traffic sign? (Data Science Interview Question asked at Latent View Analytics)

20)  If you could get the dataset on any topic of interest, irespective of the collection methods or resources then how would the dataset look like and what will you do with it. (Data Scientist Interview Question asked at CKM Advisors)

21) Given n samples from a uniform distribution [0,d], how will you estimate the value of d? (Data Scientist Interview Question asked at Spotify)

22) How will you tune a Random Forest? (Data Science Interview Question asked at Instacart).

23) Tell us about a project where you have extracted useful information from a large dataset. Which machine learning algorithm did you use for this and why? (Data Scientist Interview Question asked at Greenplum)

24) What is the difference between Z test and T test ? (Data Scientist Interview Questions asked at Antuit)

25) What are the different models you have used for analysis and what were your inferences? (Data Scientist Interview Questions asked at Cognizant)

26) Given the title of a product, identify the category and sub-category of the product. (Data Scientist interview question asked at Delhivery)

27) What is the difference between machine learning and deep learning? ( Data Scientist Interview Question asked at InfoObjects)

28) What are the different parameters in ARIMA models ? (Data Science Interview Question asked at Morgan Stanley)

29) What are the optimisations you would consider when computing the similarity matrix for a large dataset? (Data Science Interview questions asked at MakeMyTrip)

30) Use Python programming language to implement a toolbox with specific image processing tasks.(Data Science Interview Question asked at Intuitive Surgical)

31) Why do you use Random Forest instead of a simple classifier for one of the classification problems ? (Data Science Interview Question asked at Audi)

32) What is an n-gram? (Data Science Interview Question asked at Yelp)

33) What are the problems related to Overfitting and Underfitting  and how will you deal with these ? (Data Science Interview Question asked at Tiger Analytics)

34) Given a MxN dimension matrix with each cell containing an alphabet, find if a string is contained in it or not.(Data Science Interview Question asked at Tiger Analytics)

35) How do you "Group By" in R programming language without making use of any package ? (Data Scientist Interview Question asked at OLX)

36) List 15 features that you will make use of to build a classifier for OLX website.(Data Scientist Interview Question asked at OLX)

37) How will you build a caching system using an advanced data structure like hashmap ? (Data Scientist Interview Question asked at OLX)

38) How to reverse strings that have changing positions ? (Data Scientist Interview Question asked at Tiger Analytics)

39) How do you select a cricket team ? (Data Scientist Interview Question asked at Quantiphi)

40) What is the difference between trees and random forest ? (Data Scientist Interview Question asked at Salesforce)

If you are asked questions like what is your favourite leisure activity? Or something like what is that you like to do for fun?  Most of the people often tend to answer that they like to read programming books or do coding thinking that this is what they are supposed to say in a technical interview. Is this something you really do it for fun? A key point to bear in mind that the interviewer is also a person and interact with them as a person naturally. This will help the interviewer see you as an all-rounder who can visualize the company’s whole vision and not just view business problems from an academic viewpoint.

What is the biggest data set that you processed, and how did you process it, what were the results?
Tell me two success stories about your analytic or computer science projects? How was lift (or success) measured?
What is: lift, KPI, robustness, model fitting, design of experiments, 80/20 rule?
What is: collaborative filtering, n-grams, map reduce, cosine distance?
How to optimize a web crawler to run much faster, extract better information, and better summarize data to produce cleaner databases?
How would you come up with a solution to identify plagiarism?
How to detect individual paid accounts shared by multiple users?
Should click data be handled in real time? Why? In which contexts?
What is better: good data or good models? And how do you define "good"? Is there a universal good model? Are there any models that are definitely not so good?
What is probabilistic merging (AKA fuzzy merging)? Is it easier to handle with SQL or other languages? Which languages would you choose for semi-structured text data reconciliation? 
How do you handle missing data? What imputation techniques do you recommend?
What is your favorite programming language / vendor? why?
Tell me 3 things positive and 3 things negative about your favorite statistical software.
Compare SAS, R, Python, Perl
What is the curse of big data?
Have you been involved in database design and data modeling?
Have you been involved in dashboard creation and metric selection? What do you think about Birt?
What features of Teradata do you like?
You are about to send one million email (marketing campaign). How do you optimze delivery? How do you optimize response? Can you optimize both separately? (answer: not really)
Toad or Brio or any other similar clients are quite inefficient to query Oracle databases. Why? How would you do to increase speed by a factor 10, and be able to handle far bigger outputs? 
How would you turn unstructured data into structured data? Is it really necessary? Is it OK to store data as flat text files rather than in an SQL-powered RDBMS?
What are hash table collisions? How is it avoided? How frequently does it happen?
How to make sure a mapreduce application has good load balance? What is load balance?
Examples where mapreduce does not work? Examples where it works very well? What are the security issues involved with the cloud? What do you think of EMC's solution offering an hybrid approach - both internal and external cloud - to mitigate the risks and offer other advantages (which ones)?
Is it better to have 100 small hash tables or one big hash table, in memory, in terms of access speed (assuming both fit within RAM)? What do you think about in-database analytics?
Why is naive Bayes so bad? How would you improve a spam detection algorithm that uses naive Bayes?
Have you been working with white lists? Positive rules? (In the context of fraud or spam detection)
What is star schema? Lookup tables? 
Can you perform logistic regression with Excel? (yes) How? (use linest on log-transformed data)? Would the result be good? (Excel has numerical issues, but it's very interactive)
Have you optimized code or algorithms for speed: in SQL, Perl, C++, Python etc. How, and by how much?
Is it better to spend 5 days developing a 90% accurate solution, or 10 days for 100% accuracy? Depends on the context?
Define: quality assurance, six sigma, design of experiments. Give examples of good and bad designs of experiments.
What are the drawbacks of general linear model? Are you familiar with alternatives (Lasso, ridge regression, boosted trees)?
Do you think 50 small decision trees are better than a large one? Why?
Is actuarial science not a branch of statistics (survival analysis)? If not, how so?
Give examples of data that does not have a Gaussian distribution, nor log-normal. Give examples of data that has a very chaotic distribution?
Why is mean square error a bad measure of model performance? What would you suggest instead?
How can you prove that one improvement you've brought to an algorithm is really an improvement over not doing anything? Are you familiar with A/B testing?
What is sensitivity analysis? Is it better to have low sensitivity (that is, great robustness) and low predictive power, or the other way around? How to perform good cross-validation? What do you think about the idea of injecting noise in your data set to test the sensitivity of your models?
Compare logistic regression w. decision trees, neural networks. How have these technologies been vastly improved over the last 15 years?
Do you know / used data reduction techniques other than PCA? What do you think of step-wise regression? What kind of step-wise techniques are you familiar with? When is full data better than reduced data or sample?
How would you build non parametric confidence intervals, e.g. for scores? (see the AnalyticBridge theorem)
Are you familiar either with extreme value theory, monte carlo simulations or mathematical statistics (or anything else) to correctly estimate the chance of a very rare event?
What is root cause analysis? How to identify a cause vs. a correlation? Give examples.
How would you define and measure the predictive power of a metric?
How to detect the best rule set for a fraud detection scoring technology? How do you deal with rule redundancy, rule discovery, and the combinatorial nature of the problem (for finding optimum rule set - the one with best predictive power)? Can an approximate solution to the rule set problem be OK? How would you find an OK approximate solution? How would you decide it is good enough and stop looking for a better one?
How to create a keyword taxonomy?
What is a Botnet? How can it be detected?
Any experience with using API's? Programming API's? Google or Amazon API's? AaaS (Analytics as a service)?
When is it better to write your own code than using a data science software package?
Which tools do you use for visualization? What do you think of Tableau? R? SAS? (for graphs). How to efficiently represent 5 dimension in a chart (or in a video)?
What is POC (proof of concept)?
What types of clients have you been working with: internal, external, sales / finance / marketing / IT people? Consulting experience? Dealing with vendors, including vendor selection and testing?
Are you familiar with software life cycle? With IT project life cycle - from gathering requests to maintenance? 
What is a cron job? 
Are you a lone coder? A production guy (developer)? Or a designer (architect)?
Is it better to have too many false positives, or too many false negatives?
Are you familiar with pricing optimization, price elasticity, inventory management, competitive intelligence? Give examples. 
How does Zillow's algorithm work? (to estimate the value of any home in US)
How to detect bogus reviews, or bogus Facebook accounts used for bad purposes?
How would you create a new anonymous digital currency?
Have you ever thought about creating a startup? Around which idea / concept?
Do you think that typed login / password will disappear? How could they be replaced?
Have you used time series models? Cross-correlations with time lags? Correlograms? Spectral analysis? Signal processing and filtering techniques? In which context?
Which data scientists do you admire most? which startups?
How did you become interested in data science?
What is an efficiency curve? What are its drawbacks, and how can they be overcome?
What is a recommendation engine? How does it work?
What is an exact test? How and when can simulations help us when we do not use an exact test?
What do you think makes a good data scientist?
Do you think data science is an art or a science?
What is the computational complexity of a good, fast clustering algorithm? What is a good clustering algorithm? How do you determine the number of clusters? How would you perform clustering on one million unique keywords, assuming you have 10 million data points - each one consisting of two keywords, and a metric measuring how similar these two keywords are? How would you create this 10 million data points table in the first place?
Give a few examples of "best practices" in data science.
What could make a chart misleading, difficult to read or interpret? What features should a useful chart have?
Do you know a few "rules of thumb" used in statistical or computer science? Or in business analytics?
What are your top 5 predictions for the next 20 years?
How do you immediately know when statistics published in an article (e.g. newspaper) are either wrong or presented to support the author's point of view, rather than correct, comprehensive factual information on a specific subject? For instance, what do you think about the official monthly unemployment statistics regularly discussed in the press? What could make them more accurate?
Testing your analytic intuition: look at these three charts. Two of them exhibit patterns. Which ones? Do you know that these charts are called scatter-plots? Are there other ways to visually represent this type of data?
You design a robust non-parametric statistic (metric) to replace correlation or R square, that (1) is independent of sample size, (2) always between -1 and +1, and (3) based on rank statistics. How do you normalize for sample size? Write an algorithm that computes all permutations of n elements. How do you sample permutations (that is, generate tons of random permutations) when n is large, to estimate the asymptotic distribution for your newly created metric? You may use this asymptotic distribution for normalizing your metric. Do you think that an exact theoretical distribution might exist, and therefore, we should find it, and use it rather than wasting our time trying to estimate the asymptotic distribution using simulations? 
More difficult, technical question related to previous one. There is an obvious one-to-one correspondence between permutations of n elements and integers between 1 and n! Design an algorithm that encodes an integer less than n! as a permutation of n elements. What would be the reverse algorithm, used to decode a permutation and transform it back into a number? Hint: An intermediate step is to use the factorial number system representation of an integer. Feel free to check this reference online to answer the question. Even better, feel free to browse the web to find the full answer to the question (this will test the candidate's ability to quickly search online and find a solution to a problem without spending hours reinventing the wheel).  
How many "useful" votes will a Yelp review receive? My answer: Eliminate bogus accounts (read this article), or competitor reviews (how to detect them: use taxonomy to classify users, and location - two Italian restaurants in same Zip code could badmouth each other and write great comments for themselves). Detect fake likes: some companies (e.g. FanMeNow.com) will charge you to produce fake accounts and fake likes. Eliminate prolific users who like everything, those who hate everything. Have a blacklist of keywords to filter fake reviews. See if IP address or IP block of reviewer is in a blacklist such as "Stop Forum Spam". Create honeypot to catch fraudsters.  Also watch out for disgruntled employees badmouthing their former employer. Watch out for 2 or 3 similar comments posted the same day by 3 users regarding a company that receives very few reviews. Is it a brand new company? Add more weight to trusted users (create a category of trusted users).  Flag all reviews that are identical (or nearly identical) and come from same IP address or same user. Create a metric to measure distance between two pieces of text (reviews). Create a review or reviewer taxonomy. Use hidden decision trees to rate or score review and reviewers.
What did you do today? Or what did you do this week / last week?
What/when is the latest data mining book / article you read? What/when is the latest data mining conference / webinar / class / workshop / training you attended? What/when is the most recent programming skill that you acquired?
What are your favorite data science websites? Who do you admire most in the data science community, and why? Which company do you admire most?
What/when/where is the last data science blog post you wrote? 
In your opinion, what is data science? Machine learning? Data mining?
Who are the best people you recruited and where are they today?
Can you estimate and forecast sales for any book, based on Amazon public data? Hint: read this article.
What's wrong with this picture?
Should removing stop words be Step 1 rather than Step 3, in the search engine algorithm described here? Answer: Have you thought about the fact that mine and yours could also be stop words? So in a bad implementation, data mining would become data mine after stemming, then data. In practice, you remove stop words before stemming. So Step 3 should indeed become step 1. 
Experimental design and a bit of computer science with Lego's


Which machine learning model (classification vs. regression, for example) to use given a particular problem.
The tradeoffs between different types of classification models. Between different types of regression models.
How to go about training, testing, and validating results. Different ways of controlling for model complexity.
How to model a quantity that you can’t directly observe (using Bayesian approaches, for example, and when doing so, how to choose prior distributions).
The various numerical optimization techniques (maximum likelihood, maximum a posteriori).
What types of data are important for a particular set of business needs, how you would go about collecting that data.
Dealing with correlated features in your data set, how to reduce the dimensionality of data.
If you find yourself stumped on a question, don’t panic. It’s OK to ask for more context or a relevant example. But be prepared to talk theory as well. You need to know the field inside out to advance in it.

Brush Up Beforehand
Being able to talk fluently and confidently across the range of tools and methods of data analysis means a fair amount of study beforehand. You might find it useful to review your coursework and notes, and to go over the latest tech blogs and industry newsletters.

Udacity data engineer Krasnoshtan Dmytro prepared for his interview by making sure he had a firm grasp on:

Linear/polynomial regression
Decision trees
Dimensionality reduction
Clustering
and keeping up with Data Science Weekly and Machine Learning Mastery, as well as sharpening his skills through Hacker Rank and Kaggle Competitions.

Reviewing your past work, and continuing to hone and use those skills, can only help ground you more thoroughly in the material.

Talk about Yourself
Undoubtedly, you’ll be asked to go into some detail about a project you’ve worked on. As Katie Malone says, prospective employers always ask these questions.

This is your opportunity to demonstrate how you approach a data problem and how well you can report and share your results. Pick a project you really loved working on — your passion will underscore your presentation. Make sure you can explain:

Why you chose the model you did, given the problem you were trying to solve.
What the features of your data were.
How you tested and validated the results.
What you got out of the project.
And be able to extrapolate, talking about your skills in general, answering such questions as:

When you get a new data set, what do you do with it to see if it will suit your needs for a given project?
How do you handle big data sets — how would you start work on a project with an associated data set that was many tens of GB or larger?
Know the Company
In addition to knowledge and skill, employers are looking for individuals who will be a good fit with the company and its culture. It goes without saying that you need to do what you can to research the company you’re interviewing with, looking not only at their products, but finding out what you can about their office culture as well. Think about a few reasons (other than a steady paycheck!) you’d like to work there.

Be able to answer:

What’s a project you would want to work on at our company?
What data would you go after to start working on it?
What unique skills do you think you’d bring to the team?
If you’re able to provide a relevant sample or example, even better. According to Malone, you might be asked to do that on the fly anyway. “Not every place does this, and they usually tell you in advance if there’s a coding portion of the interview. But if there is, having a simple framework/methodology that you’re very comfortable with is essential.”

Beyond the Basics
Going through lists of questions typically heard during data science interviews by yourself won’t be as effective as talking through a few of these problems with a friend or fellow student. Mock interviews give you practice not only in organizing and verbalising your thoughts, but in doing so under some degree of pressure (though prepare yourself for the possibility of an anxiety-ridden interview!).

Reach out to your connections in the field and ask them how their own interview processes went and what they would ask if they were looking for a right-fit data analyst with your particular skill set.

Lewis Kaneshiro is a former Udacity instructor and current data scientist at Shazam. Not only has he endured grueling interviews, he is also interviewing KPCB Fellows for summer internships.

When looking for stand-out candidates, Lewis asks, What are the assumptions required for linear regression?

“Surprisingly this question has come up in multiple interviews throughout the years, and it tends to separate those who know linear models as ‘a function in R/Python’ or worse ‘a function in Excel,’ and those who can apply the models to actual data.”

Being able to confidently and capably verbalize and demonstrate (via a whiteboard) those assumptions has been a large chunk of Lewis’s interview experiences. He also hints at the importance of including graphical demonstrations of data that will violate each assumption.

“It is simple, but students who ignore these conditions will tend to blindly apply models without understanding the underlying use cases, and fail to recognize the need for normalization, skew adjustment, outlier detection, or other real-world issues. They also tend to need far too much oversight to be useful in an actual job. Students sometimes think they are being hired to apply a bunch of cool models to data, when in reality 90%+ of work is done with linear models and data normalization/validation.”




How to Prepare for a Machine Learning Interview
A comprehensive guide to a Machine Learning interview: the things you have to master to become a Machine Learning expert and pass an interview
Posted by Josh on 02-08-2018
At semanti.ca, we believe that Machine Learning is a skill that any software developer needs to have. The extent to which Machine Learning has to be mastered can vary, of course, depending on the applicative domain of the developer.

However, if you want to make building Machine Learning systems your day-to-day activity, you probably want to apply to a Machine Learning Engineer or Machine Learning Developer role. The interview process for such roles is quite hard. Partially, this is due to the absence of Machine Learning specialization in most university computer science departments.

As a consequence, the range of questions that can be asked during an interview for an ML role can vary a lot depending on a company.

In this tutorial, we gathered the most important points that are common to almost any ML interview. If you want to pass an interview, or just become a Machine Learning expert, you will find almost everything you need to know below.

Programming is the Key
Choose one programming language, master it, and be ready to answer practical questions by writing code in this language.

The lack of knowledge of a particular programming language will not be a dealbreaker: any language can be learned fast enough, but it takes years to learn to program.

We recommend focussing on learning Python. It is a de-facto standard in the Machine Learning community. However, C++, C#, Java, Kotlin, Scala, Clojure, Lua or even R could be possible alternatives. We don't recommend JavaScript, Perl, Ruby, and PHP.

We don't recommend either such languages as Matlab or SAS/SPSS, as they are proprietary and quite niche. The modern ML community is open-source oriented: the best tools and frameworks that are used in the state-of-the-art ML systems (TensorFlow, Keras, scikit-learn, numpy, scipy, and pandas) are all open source, well-documented and of an excellent stability. You can rely upon them to build your AI systems.

There's an almost infinite amount of Python code online you can inspire from to build your own Machine Learning solutions.

The most important things to know about any programming language are:

How to work with sets, lists, and dictionaries (maps) and when to use which;
How to handle exceptions;
Being capable of building specialized data structures such as linked lists, binary or prefix trees;
Being capable of using highly optimized vectorized operations instead of loops.
Linux is your OS
Modern Machine Learning ecosystem is hard to imagine without Linux. Of course, it's possible to be an effective Machine Learning engineer on Windows. Mac is also a good alternative. We recommend, however, to start learning Linux at the same time as Python. A successful modern ML Engineer is supposed to know how to work with the Linux file system, how to install Linux and Python packages, and how to move data from and to a Linux system.

Know your Machine Learning Routine
During the interview, you will most likely be asked to describe the typical routine of a machine learning project:

Data gathering (identifying the data you need, the sources you can get it from);
Data cleanup (removing noise);
Identifying and fixing missing values;
Feature engineering (transforming your examples into a vector format);
Train/validation/test split;
Choice of the algorithm;
Addressing the problems of high bias (underfitting) or high variance (overfitting);
Hyperparameter tuning;
Model deployment in production.
Know your Problems
Machine Learning problems can be separated into three major domains:

Supervised learning;
Unsupervised learning;
Reinforcement learning.
Supervised learning deals with situations when you have labeled training examples, such as messages with the label either spam or not spam. Such subproblems as classification or regression belong to this domain.

Unsupervised learning deals with cases when all your data is not annotated, but you have to find some structure in it. Such subproblems as clustering or topic modeling belong to this domain.

Reinforcement learning deals with sequential decision problems. Usually, you don't have examples, but rather an environment and an agent that can act in this environment. Every time the agent executes an action it gets a reward. The goal is to learn a policy that permits the agent to maximize the reward in the long term.

Know your Algorithms
You have to be able to explain how at least one algorithm from each Machine Learning domain works.

For supervised learning, you have to be able to explain (at least conceptually) how the following algorithms work:

Decision Trees;
Linear and Logistic Regression,
Support Vector Machines;
Perceptron;
Multilayer Perceptron (Feedforward Neural Network).
Specifically, in the SVM context, you have to be able to explain how kernels are used. In Multilayer Perceptron case you have to be able to talk about activations, loss functions, as well as forward and backward propagation.

For unsupervised learning, you have to be able to talk about such algorithms as K-Means clustering (and its difference from Expectation Maximization), Latent Dirichlet Allocation (and its difference from Latent Semantic Indexing). You have to be able to explain how to find the right number of clusters in the data or topics in the collection of documents.

For reinforcement learning, you can learn Q-learning, the algorithm most frequently cited in the literature. However, the knowledge of one of the modern deep reinforcement learning algorithms will play in your favor.

Know your Features
Feature extraction os one of the most important parts of any the machine learning project. You have to be able to explain how text or sound is converted into features, how one-hot encoding works and how you can transform continuous attributes into categorical (binning) or the other way around (embedding).

Know How to Visualize
In Machine Learning, data is often very high-dimensional. Visualizing data points that have more than three dimensions can be challenging for humans.

You have to know several dimensionality reduction algorithms and be able to explain how they work and how they are different from one another. We recommend to master at least these three algorithms:

Principal Component Analysis,
t-SNE, and
UMAP.
Know What to Do When Something Goes Wrong
Be able to answer the following questions:

You train your model but it fails to predict the training data correctly. What would you do?

Possible solutions include:

The data is possibly non-linearly separable, so use kernels;
The data is probably too high dimensional, so a dimensionality technique, such as PCA or UMAP, could be used;
The features are not informative enough, so add additional features or combine the existing ones into meta-features;
The neural network is too small, so increase the number of units or layers.
You train your model but it fails to predict the test data correctly. What would you do?

Possible solutions include:

If the training data is not predicted correctly either, then first make sure it is predicted correctly, then fix the issue with the test data.
If the training data is predicted correctly, then you can try:
use regularization (L1, L2, dropout);
get more training data (even if labeling more examples by hand).
Know how to Deploy your Model
As part of a bigger application, the Machine Learning model can be deployed in various ways. The most frequently used ones are:

Docker containers;
RESTful web services (flask or falcon).
When you deploy your model you have to be able to explain when the model consists of. In Python, usually, it's a pickle file that contains the model object itself as well as additional objects such as feature extractors, data normalizers, and dimensionality reducer).

Know Statistics
You have to be able to explain basic probability distributions and when each is applied. Among the most important ones are Normal, Poisson, Binomial, Multinomial and Uniform distributions.

Be ready to answer the following questions:

Name a probability distribution other than Normal and explain how to apply this probability?

How best to select a representative sample of search queries from 5 million?

The mean heights of men and women in a population were calculated to be mM and mW. What is the mean height of the total population?

Three friends in Seattle told you it’s rainy. Each has a probability of 1/3 of lying. What’s the probability of Seattle is rainy?

How do you detect if a new observation is an outlier?

What is the Central Limit Theorem? Why it's important?

Explain the difference between mean and median.

Define variance.

What is covariance? Where is it used?

What is the goal of A/B Testing?

What is sampling? Why we need it? What is stratified sampling?

Programming Challenges
Be ready to write the programming code on the whiteboard or on a computer. Usually, the challenges are not ML related because it would be quite long to accomplish. Here some examples of coding challenges:

Find all palindromic substrings in a given string.

How would you check if a linked list has cycles?

Merge k (in this case k=2) arrays and sort them.

Find max sum subsequence from a sequence of integers.

Create a function that checks if a word is a palindrome.

You have a ‘csv’ file with ID and Quantity columns, it doesn't fit in memory. Write a program in any language of your choice to aggregate the Quantity column.

Given a list A of objects and another list B which is identical to A except that one element is removed, find the removed element.

Given a list of integers (positive & negative), write an algorithm to find whether there’s at least a pair of integers that sum up to zero.

Other Things to Know
While we didn't mention it in the previous sections, these questions could be asked in the interview and you have to be prepared to answer them:

What are your go-to packages? (scikit-learn, xgboost, Pandas, numpy/scipy, Keras, matplotlib)

What techniques do you use to fine-tune hyperparameters? (Cross-validation, grid-search, random-search, tree of Parzen estimators, Bayesian optimization.)

How to quickly find if a text contains one of a million substrings. (Prefix trees.)

Explain word embeddings. How are they learned?

What is Stochastic Gradient Descent. Describe it in your own words?

Basically update weights in the network or the algorithm, depending on the
direction of steepest descent :

Wt+1 = Wt - alph * Grad(L)
alph = learning rate.
Grad = gradient of the loss function

Explain the difference between Gradient Descent and Stochastic Gradient Descent. When use which?

How can dropout be useful in a neural network?

Explain Batch Normalization. What benefit does it bring?

What is the use for a 1x1 convolution?

How to represent a text document for machine learning?

How to predict the next value in a time series?

How to measure the similarity of two words or two documents?

You have a search engine. How would you generate related searches for a query?

How would you suggest followers on Twitter?

What are support vectors in Support Vector Machines?

What's the difference between Logistic Regression and SVM?

When training an SVM, what value are you optimizing for?

What is an unbalanced classification problem and how to deal with it?

What is data wrangling? Explain its main steps.

What would you do to summarize a Twitter feed?

Explain the problem of vanishing gradient. How to deal with it?

Explain the problem of exploding gradient. How to deal with it?

Explain the need of the bias term.

Explain the difference between unsupervised, semi-supervised and self-supervised learning.

How Generative Adversarial Neural Networks (GANs) work?

What is an auto-encoder? Why do we "auto-encode" something?

What is a variational auto-encoder? Where can it be useful?

When do we use sigmoid for an output function? What is the problem with sigmoid during backpropagation?

What is transfer learning? How to do it in neural networks?

How to combine two different kinds of input (ex: image and text or sequence and vector) in a neural network?

How to make a neural network predict two different kinds of output?

How does a logistic regression model know what the optimal coefficients are?

Why use feature selection? What are the main techniques of feature selection?

Explain ensemble learning. What techniques do you know?

Why are ensemble methods superior to individual models?

Explain bagging.

Explain boosting.

How to combine predictions of several different learning algorithms?

Explain the difference between parametric and non-parametric models?

Why SVM doesn't give a probabilistic output? Can we make transform it to get probabilities?

How to build a multiclass classifier? Name two different strategies.

How to build a multilabel classifier?

What’s the trade-off between bias and variance?

How is KNN different from k-means?

What is the difference between hard and soft clustering?

Define precision and recall.

State the Bayes Theorem? Why it's so important?

How AlphaGo/AlphaZero work?

Why Naive Bayes is called "naive"?

What’s the difference between a generative and discriminative model?

How do you handle missing or corrupted data in a dataset?

How to measure the difference between two probability distributions?

What is cross-entropy? How is it useful in Machine Learning?

How to detect outliers?

Explain the difference between a test set and a validation set.

If you know the answers to most of the above questions, we are sure that you are well-equipped to pass a Machine Learning interview. Let us know if this tutorial helped you to pass an interview. Also, let us know if you think that some important aspect of an ML job is missing in our tutorial.

And do not be discouraged if you don't succeed the first time. The key to success is constant learning and persistence!

Read recent posts in our blog or subscribe to our RSS feed.

Found a mistyping or an inconsistency in the text? Let us know and we will improve it.

Like it? Share it!

    

Home · Blog · Pricing · About · Contact

 
 +1 646 9050250
 human@semanti.ca

   1. Statistics

Statistical computing is the process through which data scientists take raw data and create predictions and models backed by the data. Without an advanced knowledge of statistics it is difficult to succeed as a data scientist – accordingly it is likely a good interviewer will try to probe your understanding of the subject matter with statistics-oriented data science interview questions. Be prepared to answer some fundamental statistics questions as part of your data science interview. 

Here are examples of rudimentary statistics questions we’ve found:

What is the Central Limit Theorem and why is it important?
What is sampling? How many sampling methods do you know?
What is the difference between Type I vs Type II error?
What is linear regression? What do the terms P-value, coefficient, R-Squared value mean? What is the significance of each of these components?
What are the assumptions required for linear regression? -- There are four major assumptions: 1. There is a linear relationship between the dependent variables and the regressors, meaning the model you are creating actually fits the data, 2. The errors or residuals of the data are normally distributed and independent from each other, 3. There is minimal multicollinearity between explanatory variables, and 4. Homoscedasticity. This means the variance around the regression line is the same for all values of the predictor variable.
What is a statistical interaction?
What is selection bias?
What is an example of a dataset with a non-Gaussian distribution?
What is the Binomial Probability Formula?
2. Programming

To test your programming skills, employers will ask two things during their data science interview questions: they’ll ask how you would solve programming problems in theory without writing out the code, and then they will also offer whiteboarding exercises for you to code on the spot. 

2.1 General

With which programming languages and environments are you most comfortable working?
What are some pros and cons about your favorite statistical software?
Tell me about an original algorithm you’ve created.
Describe a data science project  in which you worked with a substantial programming component. What did you learn from that experience?
Do you contribute to any open source projects?
How would you clean a dataset in (insert language here)?
Tell me about the coding you did during your last project?
2.2 Big Data

What are the two main components of the Hadoop Framework?
Explain how MapReduce works as simply as possible.
How would you sort a large list of numbers?
Here is a big dataset. What is your plan for dealing with outliers? How about missing values? How about transformations?
2.3 Python

What modules/libraries are you most familiar with? What do you like or dislike about them?
What are the supported data types in Python?
What is the difference between a tuple and a list in Python?
2.4 R

What are the different types of sorting algorithms available in R language? -- There are insertion, bubble, and selection sorting algorithms.
What are the different data objects in R?
What packages are you most familiar with? What do you like or dislike about them?
How do you access the element in the 2nd column and 4th row of a matrix named M?
What is the command used to store R objects in a file?
What is the best way to use Hadoop and R together for analysis?
How do you split a continuous variable into different groups/ranks in R?
Write a function in R language to replace the missing value in a vector with the mean of that vector.
2.5 SQL

Often, SQL questions are case-based, meaning that an employer will task you with solving an SQL problem in order to test your skills from a practical standpoint. For example, you could be given a table and be asked to extract relevant data, filter and order the data as you see fit, and report your findings. 

What is the purpose of the group functions in SQL? Give some examples of group functions.
Group functions are necessary to get summary statistics of a dataset. COUNT, MAX, MIN, AVG, SUM, and DISTINCT are all group functions
Tell me the difference between an inner join, left join/right join, and union.
What does UNION do? What is the difference between UNION and UNION ALL?
What is the difference between SQL and MySQL or SQL Server?
If a table contains duplicate rows, does a query result display the duplicate values by default? How can you eliminate duplicate rows from a query result?
3. Modeling

Data modeling is where a data scientist provides value for a company. Turning data into predictive and actionable information is difficult, talking about it to a potential employer even more so. Practice describing your past experiences building models – what were the techniques used, challenges overcome, and successes achieved in the process? The group of questions below are designed to uncover that information, as well as your formal education of different modeling techniques. If you can’t describe the theory and assumptions associated with a model you’ve used, it won’t leave a good impression. 

Take a look at the questions below to practice. Not all of the questions will be relevant to your interview – you’re not expected to be a master of all techniques. The best use of these questions is to re-familiarize yourself with the modeling techniques you’ve learned in the past.

Tell me about how you designed the model you created for a past employer or client.
What are your favorite data visualization techniques?
How would you effectively represent data with 5 dimensions? 
How is kNN different from k-means clustering? -- kNN, or k-nearest neighbors is a classification algorithm, where the k is an integer describing the the number of neighboring data points that influence the classification of a given observation. K-means is a clustering algorithm, where the k is an integer describing the number of clusters to be created from the given data. Both accomplish different tasks.
How would you create a logistic regression model?
Have you used a time series model? Do you understand cross-correlations with time lags?
Explain the 80/20 rule, and tell me about its importance in model validation.
Explain what precision and recall are. How do they relate to the ROC curve? -- Recall describes what percentage of true positives are described as positive by the model. Precision describes what percent of positive predictions were correct. The ROC curve shows the relationship between model recall and specificity – specificity being a measure of the percent of true negatives being described as negative by the model. Recall, precision, and the ROC are measures used to identify how useful a given classification model is.
Explain the difference between L1 and L2 regularization methods.
What is root cause analysis?
What are hash table collisions?
What is an exact test?
In your opinion, which is more important when designing a machine learning model: Model performance? Or model accuracy?
What is one way that you would handle an imbalanced dataset that’s being used for prediction? (i.e. vastly more negative classes than positive classes.)
How would you validate a model you created to generate a predictive model of a quantitative outcome variable using multiple regression?
I have two models of comparable accuracy and computational performance. Which one should I choose for production and why?
How do you deal with sparsity?
Is it better to spend 5 days developing a 90% accurate solution, or 10 days for 100% accuracy?
What are some situations where a general linear model fails?
Do you think 50 small decision trees are better than a large one? Why?
When modifying an algorithm, how do you know that your changes are an improvement over not doing anything?
Is it better to have too many false positives, or too many false negatives? 



Which data scientists do you admire most? which startups?
How would you validate a model you created to generate a predictive model of a quantitative outcome variable using multiple regression.
Explain what precision and recall are. How do they relate to the ROC curve?
How can you prove that one improvement you've brought to an algorithm is really an improvement over not doing anything?
What is root cause analysis?
Are you familiar with pricing optimization, price elasticity, inventory management, competitive intelligence? Give examples.
What is statistical power?
Explain what resampling methods are and why they are useful. Also explain their limitations.
Is it better to have too many false positives, or too many false negatives? Explain.
What is selection bias, why is it important and how can you avoid it?
Give an example of how you would use experimental design to answer a question about user behavior.
What is the difference between "long" and "wide" format data?
What method do you use to determine whether the statistics published in an article (e.g. newspaper) are either wrong or presented to support the author's point of view, rather than correct, comprehensive factual information on a specific subject?
Explain Edward Tufte's concept of "chart junk."
How would you screen for outliers and what should you do if you find one?
How would you use either the extreme value theory, Monte Carlo simulations or mathematical statistics (or anything else) to correctly estimate the chance of a very rare event?
What is a recommendation engine? How does it work?
Explain what a false positive and a false negative are. Why is it important to differentiate these from each other?
Which tools do you use for visualization? What do you think of Tableau? R? SAS? (for graphs). How to efficiently represent 5 dimension in a chart (or in a video)?


I built a linear regression model showing 95% confidence interval. Does it mean that there is a 95% chance that my model coefficients are the true estimate of the function I am trying to approximate? (Hint: It actually means 95% of the time…)
What is a similarity between Hadoop file system and k-nearest neighbor algorithm? (Hint: ‘lazy’)
Which structure is more powerful in terms of expressiveness (i.e. it can represent a given Boolean function, accurately) — a single-layer perceptron or a 2-layer decision tree? (Hint: XOR)
And, which one is more powerful — a 2 layer decision tree or a 2-layer neural network without any activation function? (Hint: non-linearity?)
Can a neural network be used as a tool for dimensionality reduction? Explain how.

Everybody maligns and belittles the intercept term in a linear regression model. Tell me one of its utilities. (Hint: noise/garbage collector)
LASSO regularization reduces coefficients to exact zero. Ridge regression reduces them to very small but non-zero value. Can you explain the difference intuitively from the plots of two simple function|x| and x²? (Hint: Those sharp corners in the |x| plot)
Let’s say that you don’t know anything about the distribution from which a data set (continuous valued numbers) came and you are forbidden to assume that it is Normal Gaussian. Show by simplest possible arguments that no matter what the true distribution is, you can guarantee that ~89% of the data will lie within +/- 3 standard deviations away from the mean (Hint: Markov’s Ph.D. adviser)
Majority of machine learning algorithms involve some kind of matrix manipulation like multiplication or inversion. Give a simple mathematical argument why a mini-batch version of such ML algorithm might be computationally more efficient than a training with full data set. (Hint: Time complexity of matrix multiplication…)
Don’t you think that a time series is a really simple linear regression problem with only one response variable and a single predictor — time? What’s the problem with a linear regression fit (not necessarily with a single linear term but even with polynomial degree terms) approach in case of a time series data? (Hint: Past is an indicator of future…)

Show by simple mathematical argument that finding the optimal decision trees for a classification problem among all the possible tree structures, can be an exponentially hard problem.(Hint: How many trees are there in the jungle anyway?)
Both decision trees and deep neural networks are non-linear classifier i.e. they separates the space by complicated decision boundary. Why, then, it is so much easier for us to intuitively follow a decision tree model vs. a deep neural network?
Back-propagation is the workhorse of deep learning. Name a few possible alternative techniques to train a neural network without using back-propagation. (Hint: Random search…)
Let’s say you have two problems — a linear regression and a logistic regression (classification). Which one of them is more likely to be benefited from a newly discovered super-fast large matrix multiplication algorithm? Why? (Hint: Which one is more likely to use a matrix manipulation?)
What is the impact of correlation among predictors on principal component analysis? How can you tackle it?

You are asked to build a classification model about meteorites impact with Earth (important project for human civilization). After preliminary analysis, you get 99% accuracy. Should you be happy? Why not? What can you do about it? (Hint: Rare event…)
Is it possible capture the correlation between continuous and categorical variable? If yes, how?
If you are working with gene expression data, there are often millions of predictor variables and only hundreds of sample. Give simple mathematical argument why ordinary-least-square is not a good choice for such situation if you to build a regression model. (Hint: Some matrix algebra…)
Explain why k-fold cross-validation does not work well with time-series model. What can you do about it? (Hint: Immediate past is a close indicator of future…)
Simple random sampling of training data set into training and validation set works well for the regression problem. But what can go wrong with this approach for a classification problem? What can be done about it? (Hint: Are all classes prevalent to the same degree?)

Which is more important to you – model accuracy, or model performance?
Imagine your data set is known to be linearly separable and you have to guarantee the convergence and maximum number of iterations/steps of your algorithm (due to computational resource reason). Would you choose gradient descent in this case? What can you choose? (Hint: Which simple algorithm provides guarantee of finding solution?)
Let’s say you have a extremely small memory/storage. What kind of algorithm would you prefer — logistic regression or k-nearest neighbor? Why? (Hint: Space complexity)
To build a machine learning model initially you had 100 data points and 5 features. To reduce bias, you doubled the features to include 5 more variables and collected 100 more data points. Explain if this is a right approach? (Hint: There is a curse on machine learning. Have you heard about it?)

If you have any other fun ML question or ideas to share, please contact the author here.
Good questions are hard to generate and they give rise to curiosity and force one to think deeply.
By asking funny and interesting question, you make the learning experience enjoyable
and enriching at the same time. Hope you enjoyed this attempt of doing that.

SQL queries, learn SQL

