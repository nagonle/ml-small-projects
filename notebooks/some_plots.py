#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 nicolas <nicolas@laptop>
#
# Distributed under terms of the MIT license.

"""
Support Vector Machine
======================

Cost function plots.
"""
import matplotlib.pyplot as plt
from math import log, exp
X = []
y = []

for z in range(-5, 6):
    y.append(-log(1 - 1/(1 + exp(-z))))
    X.append(z)
    if (z == -1.0):
        print z, '=>', -log(1 - 1/(1 + exp(-z)))
plt.plot(X, y)
plt.show()
print y


# Logistic Regression Cost Function
# =================================

# 0 < z < 1
import numpy as np
for z in np.arange(0.1, 1, 0.05):
    print z,

y0 = -log(1 - z)
y1 = -log(z)
