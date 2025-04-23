# --------------------------------------------------
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/KNRscore
# Licence     : MIT
# --------------------------------------------------

import flameplot
print(flameplot.__version__)

# %% Import class
from sklearn import (manifold, decomposition)
import pandas as pd
import numpy as np
from flameplot import flameplot

# Load mnist example data
X, y = flameplot.import_example()
# PCA: 50 PCs
X_pca_50 = decomposition.TruncatedSVD(n_components=50).fit_transform(X)
# tSNE: 2D
X_tsne = manifold.TSNE(n_components=2, init='pca').fit_transform(X)
# Compare PCA(50) vs. tSNE
scores = flameplot.compare(X_pca_50, X_tsne, n_steps=25)

# Plot
fig, ax = flameplot.plot(scores, xlabel='PCA (50d)', ylabel='tSNE (2d)')
fig, ax = flameplot.scatter(X_pca_50[:,0], X_pca_50[:,1])

