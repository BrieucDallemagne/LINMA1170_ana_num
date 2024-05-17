import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

# filename = 'FD.jpg'
# img = plt.imread(filename).astype(np.uint8)
# red, green, blue = img[:,:,0], img[:,:,1], img[:,:,2]

# database = pd.DataFrame()

# def compress_image(image, color, k=2400):
#     U, S, Vh = np.linalg.svd(image, full_matrices=False)
#     database[f"S_{color}"] = S[:k]
#     database[f"U_{color}"] = np.reshape(U[:,:k], )
#     #for i in range(2400) :
#     #    database[f"U_{color}_{i}"] = U[i,:k]
#     #    database[f"Vh_{color}_{i}"] = Vh[:k,i]
#     return U, S, Vh
#     # #wite those 3 matrix in a data.txt file
#     # np.savetxt('data.txt', compressed_U, fmt='%f')
#     # np.savetxt('data.txt', compressed_Vt, fmt='%f')
#     # np.savetxt('data.txt', S1, fmt='%f')
#     # #compressed_channel = np.dot(compressed_U, np.dot(np.diag(S1), compressed_Vt))
#     # #return compressed_channel

# with open ('test.txt', 'w') as file :
#     U, S, Vh = compress_image(blue, color='blue')
#     for line in U :
#         for val in line : file.write(str(val))
#     for val in S : file.write(str(val))
#     for line in Vh :
#         for val in line : file.write(str(val))
# compress_image(green, color='green')
# compress_image(red, color='red')

# database.to_csv("test.csv")

# compressed_blue = np.clip(compressed_blue, 0, 255).astype(np.uint8)
# compressed_green = np.clip(compressed_green, 0, 255).astype(np.uint8)
# compressed_red = np.clip(compressed_red, 0, 255).astype(np.uint8)

# compressed_image = np.dstack((compressed_red, compressed_green, compressed_blue))

# plt.imshow(compressed_image)
# plt.show()
# plt.savefig('outputs/test.jpg')
# #plt.imwrite('outputs/compressed_image.jpg', compressed_image)

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
    parser.add_argument('-f', '--file', type=str, help="Fichier avec l'image à compresse")
    parser.add_argument('-o', '--output_file', type=str, help="Nom du fichier de l'image compressée", default=None)
    parser.add_argument('-k', '--components', type=int, help="Nombre de composantes principales à conserver", default=40)
    args = parser.parse_args()
    compress_image(args.file, args.output_file, args.components)