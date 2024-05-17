import matplotlib.pyplot as plt
import numpy as np

x = [1,2,5,10,50,100,200,500,900,1000,1500,2000,2400]
y = [2799.7081703278072, 2069.7135527711275, 1461.0437180080999, 1223.8825878398832,
      881.9588931637769, 723.3723772747313, 540.7711293888568, 279.2306274396062, 135.26584827464634,
        113.25120513069803, 43.080704679397016, 14.660763619904662, 3.7183331104726415e-11]


def plotCPVE(x,y):
    plt.plot(x,y)
    plt.xlabel('nombre de valeurs singulières gardées')
    plt.ylabel('CPVE')
    plt.title('CPVE en fonction du nombre de valeurs singulières gardées')
    plt.savefig('img/plotCPVE.pdf')
    plt.show()
    return

plotCPVE(x,y)