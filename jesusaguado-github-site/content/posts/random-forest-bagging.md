---
title: "Random Forest Bagging"
date: 2023-07-03T10:49:12+02:00
math: true
draft: true
---

# Bagging and Ranfom Forests

In this post I want to develop the path started in [the previous one](https://jesusaguado.github.io/posts/decision-tree/)
by analyzing how to enrich our decision tree models. This will be
an introduction to the technique of bagging models and to
Random Forest. Both are prime examples of ensembles, in which
one uses simple building blocks (also called *weak learners*) to
construct a more powerful model.

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

This process is motivated by the fact that if we have 
identically distributed and independent random variables
$Z_1,\dots, Z_r$, the sample average $\bar{Z} = \frac{1}{r} \sum_{i=1}^r Z_i$
has variance $\text{var}(\bar{Z}) = \frac{\text{var}(Z_i)}{r}$,
and thus we hope that in building our average model we will decrease the variance
of our predictions!

