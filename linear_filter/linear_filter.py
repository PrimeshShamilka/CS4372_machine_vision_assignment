"""
Linear filter implementation in python from scrath
Author: Primesh Pathirana
Date: 31/08/2021
"""

import cv2

# read filter mask 
filter_mask_arr = []
print ("Enter filter mask matrix row-wise (seperate by space): ")
for i in range(3):
    row = list(map(int, input().split()))
    # assert row size is 3
    assert len(row) == 3
    filter_mask_arr.append(row)

print ("Place lena_gray.png image in the same folder")

# read image
src = cv2.imread('lena_gray.png', 0)
# convert image to matrix
img_mat = []
filtered_empty = []
for i in range(0, src.shape[0]):
    row = []
    row_empty = []
    for j in range(0, src.shape[1]):
        pixel = src.item(i, j)
        row.append(pixel)
        row_empty.append(0)
    img_mat.append(row)
    filtered_empty.append(row_empty)


image = img_mat
kernel = filter_mask_arr
kernel_sum = 9

N = len(image)
M = len(image[0])

# fetch the dimensions for iteration over the pixels and weights
# heigt --> n_rows(y), width --> n_cols(x)
i_width, i_height = len(image[0]), len(image)
k_width, k_height = 3, 3


# Iterate over each (x, y) pixel in the image 
def linear_filter(kernel, image, filtered_empty, edge_param):

    filtered = filtered_empty

    for y in range (i_height):
        for x in range (i_width):

            # Iterate over the kernel
            weighted_pixel_sum = 0
            for ky in range(-(k_height // 2), k_height - 1):
                for kx in range(-(k_width // 2), k_width - 1):
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

print(linear_filter(kernel, image, filtered_empty, 'O'))
