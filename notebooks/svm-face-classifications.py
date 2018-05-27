import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()

print "DESCR"
print faces.DESCR
print "images.shape"
print faces.images.shape
print "data.shape"
print faces.data.shape
print "target.shape"
print faces.target.shape
