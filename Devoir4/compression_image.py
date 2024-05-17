#################################################
# compression_image.py
# Authors : Charles Van Hees and Brieuc Dallemagne
#
# This script compresses an image using the SVD method.
#################################################

import matplotlib.pyplot as plt
import numpy as np
import argparse

def apply_SVD(image, k=50) :
    """
    ARGS:
    image : np.array, image to compress (in the form of a 2D np.array)
    k : int, number of singular values to keep

    RETURNS:
    np.array, compressed image (in the form of a 2D np.array)
    """
    U, S, Vh = np.linalg.svd(image, full_matrices=False)
    return U[:,:k] @ np.diag(S[:k]) @ Vh[:k,:]

def compress_image(filename, output_file, k=50) :
    """
    ARGS:
    filename : str, name of the file with the image to compress
    output_file : str, name of the file of the compressed image
    k : int, number of singular values to keep

    RETURNS:
    None
    """
    img = plt.imread(filename).astype(np.uint8)
    red, green, blue = img[:,:,0], img[:,:,1], img[:,:,2]

    blue_data = np.clip(apply_SVD(blue, k=k), 0, 255).astype(np.uint8)
    green_data = np.clip(apply_SVD(green, k=k), 0, 255).astype(np.uint8)
    red_data = np.clip(apply_SVD(red, k=k), 0, 255).astype(np.uint8)

    new_image = np.dstack((red_data, green_data, blue_data))
    plt.imshow(new_image)
    plt.axis('off')
    if output_file == None : plt.show()
    else : plt.savefig(output_file, bbox_inches = 'tight', transparent=True, pad_inches = 0)

if __name__ == '__main__' :
    #parse arguments and call compress_image
    parser = argparse.ArgumentParser(description="Compression d'image")
    parser.add_argument('-f', '--file', type=str, help="Nom du fichier avec l'image à compresse")
    parser.add_argument('-o', '--output_file', type=str, help="Nom du fichier de l'image compressée", default=None)
    parser.add_argument('-k', '--components', type=int, help="Nombre de composantes principales à conserver", default=50)
    args = parser.parse_args()
    compress_image(args.file, args.output_file, args.components)
