# -------------------------------------------------------------------------
#
# PYTHON for DUMMIES 19-20
# Problème 4
#
# Script de test
#  Vincent Legat
#
# -------------------------------------------------------------------------
# 

from numpy import *
from bspline import bspline

#
# -1- Approximation d'un rectangle :-)     
#

X = [0,3,3,0]
Y = [0,0,2,2]
t = linspace(0,len(X),len(X)*100 + 1)
      
x,y = bspline(X,Y,t)

#
# -2- Un joli dessin :-)
#

import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rcParams['toolbar'] = 'None'

fig = plt.figure("Approximation avec des B-splines")
plt.plot(X,Y,'.r',markersize=10)
plt.plot([*X,X[0]],[*Y,Y[0]],'--r')
plt.plot(x,y,'-b')
plt.axis("equal"); plt.axis("off")
plt.show()
