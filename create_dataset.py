import numpy as np
from sklearn import datasets

# Create dataset of classification task with many redundant and few
# informative features
X, y = datasets.make_classification(n_samples=150, n_features=4, n_informative=4, n_redundant=0, random_state=42, n_classes=3, hypercube=True, n_clusters_per_class=3, class_sep=1.3)

print X
print y
