---
title: "Decision Trees"
date: 2023-05-30T10:22:10+02:00
draft: false
math: true
---
This post will introduce decision tree classifiers. You can follow
along together with the actual implementation in python with 
the `scikit-learn` package.

You can follow the code [here](/docs/decision-tree.html)
or download it in Jupyter notebook format together
with the dataset [in this zipped file](/docs/decision-tree-files.zip).


## 1. What are decision trees?

A decision tree is a statistical or machine learning model geared towards
classification or regression. We will focus in the classification problem,
and thus we will assume that we have a dataset registering the values
of $N$ attributes $(x_1,\dots,x_N)$ and a target variable $y$ which is categorical,
i.e. it takes values in a finite set.

For example, and this is detailed in the attached Jupyter notebook file,
we can consider a dataset of medical patients in which we register the following
attributes `Age`, `Sex`, blood pressure `BP`, `Cholesterol` and sodium
to potassium ratio `Na_to_K`, together with the target variable of interest
which register which drug worked best on that patient, named `Drug`, out
of five different ones. Let's discuss these variables:

1. `Age`, `Na_to_K`: these are numerical attributes taking continuous values.
2. `Cholesterol`, `BP`: these are categorical variables taking the values `HIGH`, `NORMAL` and `LOW`.
3. `Sex`: categorical variable with values `F` and `M`.
4. `Drug`: categorical variable with values `DrugA`, `DrugB`, `DrugC`, `DrugX` and  `DrugY`.

Categorical variables are best encoded in numbers, and one can do so in several ways.
For the purposes of building a decision tree it is enough to map for example the
values {`HIGH`, `NORMAL`,`LOW`} to {1,2,3} for the blood pressure and cholesterol
variables, and similarly for other categorical variables.
A decision tree for this classification problem looks as follows:

{{< img src="/pics/decision-tree/decision-tree.png" caption="Fig. 1: an already trained decision tree model for the classification problem." class="center">}}

Our decision tree model starts at the top box (*node*) and for a particular observation
(patient for which we want to predict which medication will work best) it checks
on its attributes. For example, let's say that we have a patient with
`Na_to_K = 13.4`. The model will then go down from the top node towards the
left branch, because the condition on the top node is satisfied (`Na_to_K <= 14.615`).
Otherwise it will go down towards the right. In the second leftmost node of the
second generation, the model will now check on our observation whether or not
`BP` is lower than the cut `0.5`. This procedure goes on until we end up in
a terminal node, also called *leaf* of the tree. The model will then assign
to our observation the class written on the bottom of the leaf node.

## 2. How to train/build a decision tree?

The above model is already built. In this section we explore how can approach
the problem of building such a model from some collected dataset. Suppose we 
have a dataset with $M$ observations 
$D_{\text{train}} = \left\\{x_1^i,\dots,x_N^i,y_i\right\\}_{i=1}^{M}$
of $N$ variables $x_1,\dots,x_N$ and 1 target variable $y$.

We start with the whole training dataset $D_{\text{train}}$ and select 
an attribute index $j=1,\dots,N$. We introduce another parameter $t_j$ which 
we will be using as the cut for the first node in the tree. We then partition
the dataset in two disjoint subsets:

$$ Q^{l}(j,t_j) = \\{(x,y)\in D_{\text{train}} | x_j \leq t_j \\} $$
$$ Q^{r}(j,t_j) = \\{(x,y)\in D_{\text{train}} | x_j > t_j \\} $$

Note that $D_\text{train} = Q^l(j,t_j) \sqcup Q^r(j,t_j)$. This partition
corresponds tentatively to building the first (top) node in our decision tree.
However, we have to check whether or not this partition is optimal for our
classification purposes. Think of this as if we are trying to partition the
dataset in disjoint boxes and we want these boxes, if possible, to contain
one and only one class of the target variable. We are trying to purify
or screen our partitions to "enrich" them in some class. There are various
numerical measures of how well we are doing this. One of them is 
[Shannon's entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)),
whihch in our context it is defined for one of the above sets $Q = Q^r, Q^l$
as
$$ S(Q) = \sum_{k\in\text{Classes}} -p_k \log(p_k)$$
where $p_k$ is the proportion of observations in $Q$ of class $y=k$. Note
that this is a non-negative measureand vanishes if and only if $Q$ is a pure
set containing only one class. This can be checked easily by noticing
that $\sum_{k\in\text{Classes}} p_k = 1$ and that $0 \leq p_k \leq 1$.
So the lower the entropy the better!

Now we can vary the attribute selected $j$ and the cut threshold $t_j$ so that
we can minimize the weighted entropy of the partition, namely the function
$$
\frac{|Q^l|}{|D|} S(Q^l)+
\frac{|Q^r|}{|D|} S(Q^r),
$$
where $|D|$ is the number of observations in the original dataset. This will
select, as in our example of a trained decision tree above, the `Na_to_K`
attribute ($j$) and the cut $t=14.615$.

One then proceed iteratively. Let us take the left subset $Q^r$ now as the
starting dataset overwriting $D$. Should we obtain that $D$ is already pure,
we stop there, as there's nothing we can do to improve the classification,
and we declare that node to be a tree. This is exactly what happens on 
our decision tree for the first subset $Q^r$: all those patients responded
well to `DrugY`, and that should be our prediction for these kind of observations.

On the other hand, those observations satisfying `Na_to_K<=14.615` are still
very  much mixed in classes. Further classification is need, and to continue building 
the tree we optimize for another attribute and cut.

### Some caveats: cost functions, overfitting

The above is a somewhat crude simplification of the actual implementations.
Instead of minimizing the weight sum of the two subsets one maximizes
the *information gain* which is defined as the difference in entropy with
the parent subset. One can also substitute Shannon's entropy with other measures
such as Gini impurity coefficient
$$
G(Q) = \sum_{k\in\text{Classes}} p_k (1-p_k)
$$

With a big enough decision tree one can of course achieve perfect classification
of the training by simply memorizing the classes of all observations with
a huge tree, as long as observations with the same attributes in the dataset
share the same class. This is overfitting and it is clearly bad, and we do
not expect such models to generalize well on data they have not been
trained on.

One crude way way we can try to avoid overfitting is by restricting the depth
of the tree, which in our example above was bounded above by 4. One can
also impose on nodes to contain more than some minimal number of observations
to be able to branch out, or even perform statistical significance tests
(this is known as $\chi^2$-pruning) in order to allow some attribute to
branch from a node. Other approaches add the size of tree to the overall
cost function.

There is some mathematical subtlety in the implementation of the minimization
procudure described above to select the attribute and cut that best works
for a given node, and that has to do with the fact that the function
$$f(j,t_j) =
    \frac{|Q^{\text{left}}|}{|D|} S(Q^{\text{left}})+
    \frac{|Q^{\text{right}}|}{|D|} S(Q^{\text{right}})
$$
is clearly not even continuous in the input variables. Most implementations I
have read about such as the C4.5 or ID3 implementation partition the allowed
(observed, I guess) range of the attribute $x_j$ in a discrete set 
$\\{t_j^1,\dots,t_j^n\\}$ and evaluate the cost function only on these finite
set of values. Note that $f$ not even being continuous means the minimization
problem cannot be solved via gradient descent, even if we are to cleverly
estimate the gradient discretely!

## References

1. [Scikit-learn documentation.](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
2.  Quinlan, J. R. (1986).  Induction of decision trees.  Machine learning, 1, 81-106.
