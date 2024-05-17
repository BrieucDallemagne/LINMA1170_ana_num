import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

"""
CPVE = []
for k in [1000,1500,2000,2400]:
    compressed_blue = compress_channel(blue,k)
    compressed_green = compress_channel(green,k)
    compressed_red = compress_channel(red,k)

    compressed_image = cv2.merge([compressed_blue, compressed_green, compressed_red])

    num_columns = img.shape[1]
    distances = []
    
    for j in range(num_columns):
        vector_a = img[:, j]
        vector_b = compressed_image[:, j]
        distance = np.linalg.norm(vector_a - vector_b)
        distances.append(distance)

    cpve_value = np.mean(distances)
    
    CPVE.append(cpve_value)
    print(k, cpve_value)
"""

def apply_SVD(image, k=50) :
    U, S, Vh = np.linalg.svd(image, full_matrices=False)
    return U[:,:k] @ np.diag(S[:k]) @ Vh[:k,:]

def compress_image(filename, output_file, k=50) :
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
    parser = argparse.ArgumentParser(description="Compression d'image")
    parser.add_argument('-f', '--file', type=str, help="Nom du fichier avec l'image à compresse")
    parser.add_argument('-o', '--output_file', type=str, help="Nom du fichier de l'image compressée", default=None)
    parser.add_argument('-k', '--components', type=int, help="Nombre de composantes principales à conserver", default=50)
    args = parser.parse_args()
    compress_image(args.file, args.output_file, args.components)
