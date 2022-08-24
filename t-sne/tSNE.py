import matplotlib.pyplot as pp
import numpy as np
import sklearn.datasets as datasets
import sklearn.manifold as manifold
from scipy.sparse import issparse

mnistBunch   = datasets.load_digits(n_class = 6)
imageNDArray = mnistBunch.data
labelNDArray = mnistBunch.target

tsneNDArray = manifold.TSNE(n_components = 2, init = "random", random_state = 0).fit_transform(imageNDArray)

figure, axesSubplot = pp.subplots()

axesSubplot.scatter(tsneNDArray[:, 0], tsneNDArray[:, 1], c = labelNDArray)
axesSubplot.set_xticks(())
axesSubplot.set_yticks(())

pp.show()
