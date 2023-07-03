---
title: "Random Forest Bagging"
date: 2023-07-03T10:49:12+02:00
math: true
draft: false
---

# Bagging and Ranfom Forests

In this post I want to develop the path started in [the previous one](https://jesusaguado.github.io/posts/decision-tree/)
by analyzing how to enrich our decision tree models. This will be
an introduction to the technique of bagging models and to
Random Forest. Both are prime examples of ensembles, in which
one uses simple building blocks (also called *weak learners*) to
construct a more powerful model.

I have explored this with a standard dataset regarding pricing of houses
in California. You can follow along in [this notebook](/docs/bagged-trees-vs-random-forest.html)
or directly [download the .ipynb file](/docs/bagged-trees-vs-random-forest.ipynb).

## 1. Bagging

The process of bagging or bootstrapping consists of building different $m$
different datasets $D_i$ from the original one $D$. We can then use each
of them to train a *weak learner* $\hat{f}_i(x)$ and then use these to build a more
powerful one. In the conetxt of regression one would define the model as

$$
\hat{f}(x) = \frac{1}{m} \sum_{i=1}^m \hat{f}_i(x)
$$

while in the case of classification it would be simply a majority vote.
Each datset $D_i$ is created by sampling with replacement over the original 
dataset $D$. In our context, if we have $n$ observations in $D$, we will
sample also $n$ observations, bearing in mind that we might (and most probably
will) extract some observations multiple times.

In other contexts, we can sample with replacement $n' < n$ observations for
other purposes, but for building simple ensemble models we set $n' = n$.

This process is motivated by the fact that if we have independent
identically distributed random variables
$Z_1,\dots, Z_r$, the sample average $\bar{Z} = \frac{1}{r} \sum_{i=1}^r Z_i$
has variance $\text{var}(\bar{Z}) = \frac{\text{var}(Z_i)}{r}$,
and thus we hope that in building our average model we will decrease the variance
of our predictions!

Upon creation of each dataset $D_i$ we can also consider the corresponding
*out-of-bag* sample, which is the set difference $D_i^{OOB} = D \setminus D_i$.
There is a very nice application which justifies the expense of storing this
out-of-bag samples, which is model evaluation. One can evaluate the model
as follows. Take an observation $(x,y)\in D$ and compare the actual target
value $y$ with the prediction by the subset of weak-learners that **did not
see this observation upon training**, that is, the difference
$$
y - \frac{1}{|D^{OOB}(y)|} \sum_{i\in D^{OOB}(y)} \hat{f}_i(x)
$$
where $D^{OOB}(y) = \\{ i | (x,y) \not\in D_i, 1 \leq i \leq m \\}$. On 
average, for large $n$, only around a third of the unique observations
in $D$ are in $D_i$. We can estimate the averaged model's performance as
for example the root mean square of the above differences.

## 2. Random Forest

While bagging can in principle be used for many different models, when
using regression or decision trees as weak learners there is a further 
technique that may improve the efficiency of the ensemble. Let us assume that
there is a very important feature $X_j$ that highly correlates with the
target variable $y$. Then, when building the first node of a decision tree
on some bag $D_i$, it will most probably use this particular variable for
setting up the first split in the tree.

This process occurs when growing the rest of the trees as well, and the
resulting consequence is that most trees of the ensemble will be quite similar.
This can imply some overfittinf, for if there are more important-but-not-as-much-as-$X_j$
variables, the single decision trees might tend to overlook them.

In a Random Forest this is prevented as follows: when growing a tree, one may
create a node split using only features from a randomly selected subset of
all the original ones $X_1,\dots, X_k$. In practice, one usually
selects $\sqrt{k}$ of the original $k$ features. This results in a decorrelation
of the trees, as some of them will be forced to consider first some features
that correlate slightly less than the dominant one.

## 3. Comparison

In our code example we have implemented models through bagging of regression
trees and through random forests. We have evaluated these and obtained the
following $R^2$ scores.

![M](/pics/random-forest/bagging-vs-random-forest.png)

We can conclude that for large $n$ the Random Forest approach leads to better
results. It seems as though constraining the number of features to be used
in each split growing the trees in the forest leads to less bias.
