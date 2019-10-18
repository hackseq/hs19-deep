# An interpretable machine learning framework for detecting epistatic interactions between SNPs

## Background
The majority of interesting phenotypes are polygenic in nature, but conventional methods for computing _sets_ of SNPs which are statistically associated with a particular phenotype are computationally intractable. In this project, participants will develop an open-source interpretable deep learning framework for predicting phenotypes given SNPs, and detecting epistatic interactions between these SNPs.

Deep learning has experienced an explosion in interest in the bioinformatics community owing to the abundance of both big datasets and cheap computing resources.
However, one flaw of conventional deep neural networks is that the models are often hard to interpret - it is very hard to understand why a deep neural network makes the decisions that it makes.
For problems such as image classification and speech recognition (two fields which have been revolutionized by deep learning), this isn't a huge issue because performance is more important than interpretability.
However, interpretation in the biological sciences is important for many reasons, one being that understanding the inner workings of a well-trained model might reveal hidden biological relationships in the datasets.

A few years ago, a google engineer by the name of Alexander Mordvintsec created a computer vision program called "[DeepDream](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)" that provides a way to understand the reasoning behind convolutional neural network models (a popular deep neural network architecture for image recognition) by visualizing the sort of "patterns" that the CNN looks for when making a classification.
Algorithmically, DeepDream creates images that maximize the activation of user-specified neurons in the CNN.
For example, you can use DeepDream to maximize the activation of the neuron that predicts dumbbells, and you'll find that the CNN often looks for arms holding onto dumbbells to make that prediction.
Hilarious!
Check out the [blog post](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) to find out more about DeepDream.

We'll be applying a similar idea to a neural network that predicts antibiotic resistance in _Mycobacterium tuberculosis_ isolates given their SNPs.

## Method
This project will involve two main parts:
1. Train a deep learning model to predict antibiotic resistance in tuberculosis isolates given their SNPs.
2. Implement a combinatorial optimization algorithm (e.g. simulated annealing or genetic algorithm) that chooses a set of SNPs that maximizes the activation of the antibiotic resistance prediction neuron (the final neuron in the network).

For creating and training the deep learning model, we'll follow the methodology described in this [paper](https://arxiv.org/pdf/1801.02977.pdf).
They used a stacked autoencoder to predict preterm birth classification in women given the women's SNPs.

For the combinatorial optimization algorithm, we'll either use a [simulated annealing algorithm](https://en.wikipedia.org/wiki/Simulated_annealing) or a [genetic algorithm](https://en.wikipedia.org/wiki/genetic_algorithm) to choose a set of SNPs which maximize the activation of the final prediction neuron.

### Interpretting the neural network
You may be wondering why we're maximizing the activation of the final prediction neuron, and what it means biologically for a set of SNPs to maximize the activation.
Good question!
I'll explain with some mathematical notation (bare with me).

Often times, we make the final prediction neuron of a neural network to be a softmax vector [p_1,p_2,...,p_n].
Each component p_i in the softmax vector can be interpretted as the probability that the true classification is the ith choice. For example, if the ith choice is that the tuberculosis isolate is resistant to the antibiotic rifampicin, then the neural network is telling us that it believes the probability to be p_i.

In the following paragraph, let X be the input feature vector (i.e. a vector of SNPs in our project), W is a vector of parameters (weights) of our deep learning network, and Y be the event that the current TB isolate is resistant to some antibiotic.

One way we can interpret a deep neural network is that it is a function f(X; W) that approximates a probability distribution P(Y|X), which is the true probability that the isolate is antibiotic resistant given their SNPs.
If a set of SNPs X' maximizes P(Y|X) over all other sets of SNPs X, then the biological interpretation is that it is highly probable that a tuberculosis isolate having the SNPs X' is antibiotic resistant.
If we train the neural network well, f(X; W) ~ P(Y|X).
So, if X' maximizes f(X; W), then we can interpret X' as being highly likely to cause antibiotic resistance.

## Dataset
A dataset of SNPs and antibiotic resistance classification of around 6000 tuberculosis isolates.

## Computing resources
Because deep learning requires heavy computational resources, we'll be using Amazon Web Services along with our personal computers.
Please make an AWS account before the start of the hackathon!

## Software
Please download and install
* R version 3.5 or greater
* Python 3
* [PLINK](http://zzz.bwh.harvard.edu/plink/), a GWAS software package
* [Keras](https://keras.io)
* [Pytorch](https://pytorch.org)
* [This autoencoder implementation in python](https://github.com/jatinshah/ufldl_tutorial/blob/master/stacked_autoencoder.py)
