# -------------------------------------------------------------------------
#
# PYTHON for DUMMIES 19-20
# Problème 3
#
# Solution détaillée
#  Vincent Legat
#
# -------------------------------------------------------------------------
# 

from numpy import *

#
# -1- Definition récursive de fonctions B-spline
#

def b(t,T,i,p):

  if p == 0:
    return (T[i] <= t)*(t < T[i+1])
  else:
    u  = 0.0 if T[i+p ]  == T[i]   else (t-T[i])/(T[i+p]- T[i]) * b(t,T,i,p-1)
    u += 0.0 if T[i+p+1] == T[i+1] else (T[i+p+1]-t)/(T[i+p+1]-T[i+1]) * b(t,T,i+1,p-1)
    return u

#
# -------------------------------------------------------------------------
#
# -2- Le devoir itself :-)
#     Il est possible d'implémenter l'algorithme de DeBoor pour calculer efficacement
#     et précisément les courbes.  Mais reproduire simplement le code fourni dans l'exemple
#     de la séance était le meilleur compromis entre temps consacré au devoir et note :-)
#     Oui : cela rapportait 10/10 : so easy :-)
#
#     Pour obtenir le 11eme point, il faut soit s'attaquer à l'algorithme de DeBoor, ou
#     de manière plus astucieuse, remarquer qu'on a toujours des noeuds équidistants et donc
#     que la fonctions de forme sont donc connues a priori : bien vu Miguel, Loic 
#     et quelques autres petits futés !
#     Je vous laisse le soin d'essayer d'implémenter vous-même la solution rapide :-)
#
    
def bspline(X,Y,t): 

#
# -2.1- Definition des noeuds et duplication de 3 premiers points de controle
#       pour avoir une courbe fermée
#

  T = range(-3,len(X)+4)
  X = [*X,*X[0:3]]
  Y = [*Y,*Y[0:3]]

#
# -2.2- Calcul de la courbe B-spline en évaluant les matrices d'influences de
#       tous les points de controle pour chaque abscisse...
#
  
  p = 3; n = len(T)-1  
  B = zeros((n-p,len(t)))  
  for i in range(0,n-p):
    B[i,:] = b(t,T,i,p) 
    
#
# -2.3- Produit matriciel pour obtenir x et y
#
    
  x = X @ B
  y = Y @ B  
  return x,y