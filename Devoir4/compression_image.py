import cv2
import numpy as np

filename = 'FD.jpg'
img = cv2.imread(filename)  
print(img.shape)


blue, green, red = cv2.split(img)


def compress_channel(channel):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    k = 50  
    S1 = np.zeros(k)
    S1[:k] = S[:k]
    compressed_U = U[:, :k]
    compressed_Vt = Vt[:k, :]
    #wite those 3 matrix in a data.txt file
    np.savetxt('data.txt', compressed_U, fmt='%f')
    np.savetxt('data.txt', compressed_Vt, fmt='%f')
    np.savetxt('data.txt', S1, fmt='%f')
    #compressed_channel = np.dot(compressed_U, np.dot(np.diag(S1), compressed_Vt))
    #return compressed_channel



compressed_blue = compress_channel(blue)
compressed_green = compress_channel(green)
compressed_red = compress_channel(red)

#compressed_image = cv2.merge([compressed_blue, compressed_green, compressed_red])

#cv2.imwrite('outputs/compressed_image.jpg', compressed_image)
