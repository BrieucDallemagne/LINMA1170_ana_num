import matplotlib.pyplot as plt
import numpy as np

filename = 'FD.jpg'
img = plt.imread(filename).astype(np.uint8)
plt.imshow(img)
plt.axis('off')
plt.show()
print(img.shape)


red, green, blue = img[:,:,0], img[:,:,1], img[:,:,2]


def compress_channel(channel):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    k = 20  
    S1 = np.zeros_like(S)
    S1[:k] = S[:k]
    compressed_channel = np.dot(U, np.dot(np.diag(S1), Vt))
    return compressed_channel

compressed_blue = compress_channel(blue)
compressed_green = compress_channel(green)
compressed_red = compress_channel(red)

compressed_blue = np.clip(compressed_blue, 0, 255).astype(np.uint8)
compressed_green = np.clip(compressed_green, 0, 255).astype(np.uint8)
compressed_red = np.clip(compressed_red, 0, 255).astype(np.uint8)

compressed_image = np.dstack((compressed_red, compressed_green, compressed_blue))

plt.imshow(compressed_image)
plt.show()
plt.savefig('outputs/test.jpg')
#plt.imwrite('outputs/compressed_image.jpg', compressed_image)
