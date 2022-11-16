---
title: "Sphere Geometry"
date: 2022-11-16T11:03:57+01:00
math: true
draft: false
tags: ['machine-learning','clustering','math']
---

# Geometry and analysis of spatial data on the Earth

This is a short post about the DBSCAN algorithm when applied to spatial datapoints
on the Earth. I became interested in this when trying to cluster weatherstations
in Canada.

Clustering is a technique in unsupervised machine learning that, given a numeric
dataset, aims at bundling together some observations based on coordinate similarities.
When working with abstract data, we often think of this dataset as lying on 
\\(m\\)-dimensional space \\(\mathbb{R}^m\\), where \\(m\\) is the number of variables or
columns in our data.

DBSCAN, or [Density-based spatial clustering of applications with noise](https://en.wikipedia.org/wiki/DBSCAN)
is a particular algorithm that clusters data in a noise-robust manner, allowing
for outliers. The main idea behind
it is to identify high-density areas in our data and declare these points as
belonging to the same cluster. To do this, DBSCAN goes over the following steps
together with two prescribed parameters: \\(r\\) and \\(N\\) (*radius* and
*number of neighbors*):

1. Select any point in our dataset, call it \\(x_0\\).
2. Check how many other datapoints are nearer than \\(r\\) to this \\(x_0\\).
3. If there are at least \\(N\\) of datapoints closer than \\(r\\), \\(x_0\\)
is declared to be a *core* point in the dataset.

Points that are not core but are close (nearer than \\(r\\)) to a core point
are called *border* points. The rest of them are declared to be *outliers*

Then DBSCAN searchs for connected patches of core and border points, meaning
that you can hop on from any of the points in the patch by moving from 
point to point in steps of at most length \\(r\\). These patches are the
clusters that we seek.

## On the importance of geometry

One thing you can notice is that DBSCAN's performance is dependent on several
key factors. First, you need to tune the relevant parameters \\(r, N\\). Otherwise
DBSCAN lacks a sense of the scale of density that we are looking for.

The other, more subtle aspect, is the *distance function*. Note how when we
think of abstract data, we have some inertia towards picking simply the 
Euclidean distance, from which we compute the distance of two points as

$$ d(x,y) = \sqrt{\sum_{i=1}^m (x_i - y_i)^2}$$

This is definitely **not correct** for clustering spatial data on the Earth!
To illustrate this, I generated some random data which you can see on
the next figure. This is the [equirrectangular projection](https://en.wikipedia.org/wiki/Equirectangular_projection)
where we plot geographical longitude and latitude of points.

{{< img src="/pics/sphere-dbscan/earth-blobs.png" caption="Fig. 1: some randomly generated datapoints on the Earth. Thick outline indicates the centroid of the generated blob, in the same color." class="center">}}

We can see that the DBSCAN algorithm fails at properly clustering generated nearby
data.

{{< img src="/pics/sphere-dbscan/dbscan-flat-metric.png" caption="Fig. 2: DBSCAN clustering output with 'Euclidean' distance function" class="center">}}

In the above figure, clusters 7 and 9 should really be the same, as going around
the East border should correspond to enter from the West border of the picture.
Furthermore, clusters 2, 6, 8 and 10 all touch the north-pole, meaning they
should correspond to a continuous cluster. This also happens at the south pole
with clusters 0, 1 and 4.

To solve this deficiency, I wrote a Python module to properly compute
distances on the round sphere. You can check it out at [this Github
repository](https://github.com/jesusaguado/spherics). Passing the corrected
version of the distance function to DBSCAN yields a more realistic
clustering of the data:

{{< img src="/pics/sphere-dbscan/dbscan-round-metric.png" caption="Fig. 3: DBSCAN clustering output with correct spherical distance function" class="center">}}

Here, points that are close to the poles are clustered together, and so are
the points that share latitude but differ in 360º in longitude: these are all
bundled together in cluster number 4.

Don't be foooled by the different geographical map projections! One should
never measure distance "by eye" when dealing with global maps. Objects
near the poles are closer than they appear; objects close to the
equator are farther than they seem.

## Random data on the sphere

The above cited Python module also has a nice function to generate uniformly
distributed random datapoints on spheres. Note that should you uniformly generate
random numbers for the latitude and longitude, the poles will get much
more density than they deserve, as opposite to regions to the equator.

One can solve this by generating a 3-dimensional Gaussian distribution of
points and then normalizing them, as this is a spherically symmetric
distribution. This is already implemented in the Python module.
