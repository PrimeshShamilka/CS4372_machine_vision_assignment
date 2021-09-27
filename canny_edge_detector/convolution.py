import cv2
import numpy as np

def convolution(kernel, image, edge_param=""):
    i_height = len(image)
    i_width = len(image[0])
    k_height = len(kernel)
    k_width = len(kernel[0])
    filtered = [[0 for i in range(i_width)] for i in range(i_height)]
    for y in range (i_height):
        for x in range (i_width):
            # Iterate over the kernel
            weighted_pixel_sum = 0
            m = k_height // 2
            n = k_width // 2
            for ky in range(-m, m +1):
                for kx in range(-n, m + 1):
                    pixel = 0
                    pixel_y = y - ky
                    pixel_x = x - kx
                    # Handle edge pixels
                    if (pixel_y >= 0) and (pixel_y < i_height) and (pixel_x >= 0) and (pixel_x < i_width):
                        pixel = image[pixel_y][pixel_x]
                    weight = kernel[ky + (k_height // 2)][kx + (k_width // 2)]
                    weighted_pixel_sum += pixel * weight
            filtered[y][x] = weighted_pixel_sum
    return filtered

img = cv2.imread("/media/primesh/F4D0EA80D0EA4906/ACADEMICS/MORA/Sem 7/Machine vision/canny_edge_detector/canny_edge_detector/faces_imgs/2.jpg", 0)
img_mat = []
for i in range(0, img.shape[0]):
    row = []
    for j in range(0, img.shape[1]):
        pixel = img.item(i, j)
        row.append(pixel)
    img_mat.append(row)

# kernel = [[1, 1, 1],
#           [1, 1, 1],
#           [1, 1, 1]]

kernel_size = 3
kernel = [[1]*kernel_size]*kernel_size

res = convolution(kernel, img_mat)
cv2.imwrite("res_convolution.png", np.float32(res))