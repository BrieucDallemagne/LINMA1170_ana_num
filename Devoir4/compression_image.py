import cv2
import numpy as np

filename = 'FD.jpg'
img = cv2.imread(filename) 
img.copy() 



blue, green, red = cv2.split(img)


def compress_channel(channel,k):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    S1 = np.zeros(k)
    S1[:k] = S[:k]
    compressed_U = U[:, :k]
    compressed_Vt = Vt[:k, :]
    compressed_channel = np.dot(compressed_U, np.dot(np.diag(S1), compressed_Vt))
    return compressed_channel

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




    #cv2.imwrite('outputs/compressed_image_'+str(k)+'.jpg', compressed_image)
"""
#print(CPVE)
k= 20
compressed_blue = compress_channel(blue,k)
compressed_green = compress_channel(green,k)
compressed_red = compress_channel(red,k)

compressed_image = cv2.merge([compressed_blue, compressed_green, compressed_red])

cv2.imwrite('outputs/compressed_image.jpg', compressed_image)
