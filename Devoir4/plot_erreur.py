#################################################
# plot_erreur.py
# Authors : Charles Van Hees and Brieuc Dallemagne
#
# This script computes the error for an image and plots it.
#################################################

import matplotlib.pyplot as plt
import numpy as np

# import image
img = plt.imread('inputs/FD.jpg').astype(np.uint8)

# extract channels
red, green, blue = img[:,:,0], img[:,:,1], img[:,:,2]

def compress_channel(channel, k=50) :
    """
    ARGS:
    channel : np.array, channel to compress (in the form of a 2D np.array)
    k : int, number of singular values to keep

    RETURNS:
    np.array, compressed channel (in the form of a 2D np.array)
    """
    
    U, S, Vh = np.linalg.svd(channel, full_matrices=False)
    return U[:,:k] @ np.diag(S[:k]) @ Vh[:k,:]


error = []
# loop over different values of k to compute CPVE
for k in [1,2,5,10,50,100,200,500,900,1000,1500,2000,2400]:
    compressed_blue = compress_channel(blue,k)
    compressed_green = compress_channel(green,k)
    compressed_red = compress_channel(red,k)

    compressed_image = np.dstack((compressed_red, compressed_green, compressed_blue))

    num_columns = img.shape[1]
    distances = []
    
    for j in range(num_columns):
        vector_a = img[:, j]
        vector_b = compressed_image[:, j]
        distance = np.linalg.norm(vector_a - vector_b)
        distances.append(distance)

    mean_value = np.mean(distances)
    
    error.append(mean_value)
    print(k, mean_value)
    

#données venant de la sortie de la boucle
x = [1,2,5,10,50,100,200,500,900,1000,1500,2000,2400]
#y = [2799.7081703278072, 2069.7135527711275, 1461.0437180080999, 1223.8825878398832,881.9588931637769, 723.3723772747313, 540.7711293888568, 279.2306274396062, 135.26584827464634,113.25120513069803, 43.080704679397016, 14.660763619904662, 3.7183331104726415e-11]


def ploterror(x,y):
    """
    ARGS:
    x : list, x values
    y : list, y values

    RETURNS:
    None
    """
    plt.plot(x,y)
    plt.xlabel('nombre de valeurs singulières gardées')
    plt.ylabel('error')
    plt.title('erreur en fonction du nombre de valeurs singulières gardées')
    plt.savefig('img/plot_error.pdf')
    plt.show()
    return

ploterror(x,error)