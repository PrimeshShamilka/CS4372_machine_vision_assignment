import cv2
import numpy as np
from matplotlib import pyplot as plt
def singlePixelConvolution(image,x,y,k,kernelWidth=3,kernelHeight=3):
    output = 0
    for i in range(kernelWidth):
        for j in range(kernelWidth):
            output = output + (image[x + i][y + j] * k[i][j])    
    return output       
def mean_filtering(image,kernel):
        image = wrap(image)
        kernel = get_mean_filter(kernel)
        smallWidth = len(image) - 2
        smallHeight = len(image[0]) - 2
        output = [([0]*smallHeight) for i in range(smallWidth)]
        for i in range(smallWidth):
            for j in range (smallHeight):
                output[i][j] = singlePixelConvolution(image, i, j, kernel,3,3) 
        return output 
def medain_filtering(image,kernel):
        image = wrap(image)
        kernel = get_mean_filter(kernel)
        size = len(kernel)
        smallWidth = len(image) - 2
        smallHeight = len(image[0]) - 2
        output = [([0]*smallHeight) for i in range(smallWidth)]
        for i in range(smallWidth):
            for j in range (smallHeight):
                array = []
                for n in range(size):
                    for m in range(size):
                        array.append(image[i+n][j+m])       
                array.sort()
                median = size*size//2
                output[i][j] = array[median]
        return output 

def mid_point_filtering(image,kernel):
        image = wrap(image)
        kernel = get_mean_filter(kernel)
        size = len(kernel)
        smallWidth = len(image) - 2
        smallHeight = len(image[0]) - 2
        output = [([0]*smallHeight) for i in range(smallWidth)]
        for i in range(smallWidth):
            for j in range (smallHeight):
                array = []
                for n in range(size):
                    for m in range(size):
                        array.append(image[i+n][j+m])       
                mid = (min(array) + max(array))//2
                output[i][j] = mid
        return output 

def get_mean_filter(kernel):
    kernel_size = len(kernel)
    sum_value = 0
    for raw in kernel:
        sum_value += sum(raw)   
    for i in range(kernel_size):
        for k in range(kernel_size):
            kernel[i][k] = kernel[i][k]/sum_value
    return kernel            

# def wrap(image):
#     height = len(image)  
#     width = len(image[0])
#     new  = [image[-1]] + image + [image[0]]
#     new_image=[]
#     for i in range(len(new)):
#         array = new[i]
#         new_array = [array[-1]] + array + [array[0]]
#         new_image.append(new_array)
#     return new_image

def wrap(image):
    height = len(image)  
    width = len(image[0])
    new = []
    new.append(image[-1])
    for i in range(height):
        new.append(image[i])
    new.append(image[0])
    new_image=[]
    for i in range(len(new)):
        array = new[i]
        new_array = []
        new_array.append(array[-1])
        for k in range(width):
            new_array.append(array[k])
        new_array.append(array[0])    
        new_image.append(new_array)  
    return new_image

# if _name_ == "_main_":
#     filters = ["mean","median","midpoint"]
#     filter = input('Filter (mean,median,midpoint): ')
#     if filter  not in filters:
#         raise AssertionError("message")
kernel = [[1,1,1],[1,1,1],[1,1,1]]
image = cv2.imread('lena.png',0)
print (image.shape)
out = medain_filtering(image,kernel)
# print (out.shape)
# cv2.imwrite('out.png', out)
f, axarr = plt.subplots(2,1)
axarr[0].imshow(image,cmap='gray')
axarr[1].imshow(out,cmap='gray')
# plt.imshow(image, cmap='gray')
# plt.imshow(out, cmap='gray')
plt.show()
