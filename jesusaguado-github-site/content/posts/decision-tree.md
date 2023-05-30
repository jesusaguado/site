---
title: "Decision Trees"
date: 2023-05-30T10:22:10+02:00
draft: false
---

In this post I will briefly review what a decision tree classifier is and go
over a very basic implementation of one using Python and the scikit-learn
library.

I will focus in classification rather than regression, meaning I have some
data and I want to predict *classes* of observations, meaning a discrete
variable such as the species of some animal from a finite set of options based
on either numerical attributes (e.g weight, average lifespan) or categorical
ones (e.g *does it have wings?*)

$$\int_\Omega f(x) dx$$
