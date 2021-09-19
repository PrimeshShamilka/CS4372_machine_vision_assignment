import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
       
def mean_filtering(image,kernel):
    image = wrap(image)
    kernel_sum = len(kernel)*len(kernel[0])
    size = len(kernel)
    sub_width, sub_height = len(image) - (size-1), len(image[0]) - (size-1)
    output = [([0]*sub_height) for i in range(sub_width)]
    for i in range(sub_width):
        for j in range (sub_height):
            value = 0
            for n in range(size):
                for m in range(size):
                    value = value + image[i+n][j+m]*kernel[n][m]
            output[i][j] = int(value/kernel_sum)      
    return output 

def medain_filtering(image,kernel):
    image = wrap(image)
    size = len(kernel)
    sub_width, sub_height = len(image) - (size-1), len(image[0]) - (size-1)
    output = [([0]*sub_height) for i in range(sub_width)]
    for i in range(sub_width):
        for j in range (sub_height):
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
    size = len(kernel)
    sub_width, sub_height = len(image) - (size-1), len(image[0]) - (size-1)
    output = [([0]*sub_height) for i in range(sub_width)]
    for i in range(sub_width):
        for j in range (sub_height):
            array = []
            for n in range(size):
                for m in range(size):
                    array.append(image[i+n][j+m])       
            mid = (min(array) + max(array))//2
            output[i][j] = int(mid)
    return output        

def wrap(image):
    h, w = len(image), len(image[0])
    # wrap rows
    wrapped_rows = []
    wrapped_rows.append(image[-1])
    for i in range(h):
        wrapped_rows.append(image[i])
    wrapped_rows.append(image[0])
    wrapped_image=[]
    # wrap cols
    for i in range(len(wrapped_rows)):
        array = wrapped_rows[i]
        wrapped_cols = []
        wrapped_cols.append(array[-1])
        for k in range(w):
            wrapped_cols.append(array[k])
        wrapped_cols.append(array[0])    
        wrapped_image.append(wrapped_cols)  
    return wrapped_image

def merge_channels(b,g,r):
    image = []
    for i in range(len(r)):
        row = []
        for j in range(len(r[0])):
            row.append([b[i][j],g[i][j],r[i][j]])  
        image.append(row)
    return image   


if __name__ == "__main__":
    kernel = [[1,1,1],[1,1,1],[1,1,1]]
    ext = ['jpg', 'jpeg','png']    
    files = []
    [files.extend(glob.glob('*.' + e)) for e in ext]
    images = [cv2.imread(file) for file in files]
    filenames = [file.split(".")[0] for file in files]
    for i in range(len(images)):
        image = images[i]
        filename = filenames[i]

        #split channels
        b, g, r = image[:,:,0], image[:,:,1], image[:,:,2]

        # apply mean filter
        mean_b = np.array(mean_filtering(b,kernel))
        mean_g = np.array(mean_filtering(g,kernel))
        mean_r = np.array(mean_filtering(r,kernel))
        mean_filtered_image = mean_b + mean_g + mean_r
        mean_res  = merge_channels(mean_b, mean_g, mean_r)
        cv2.imwrite(filename + "_mean_filtered.jpg", np.array(mean_res))

        # apply medain filter
        median_b = medain_filtering(b,kernel)
        median_g = medain_filtering(g,kernel)
        median_r = medain_filtering(r,kernel)
        median_res  = merge_channels(median_b, median_g, median_r)
        cv2.imwrite(filename + "_median_filtered.jpg", np.array(median_res))

        # apply midpoint filter
        mid_point_b = mid_point_filtering(b,kernel)
        mid_point_g = mid_point_filtering(g,kernel)
        mid_point_r = mid_point_filtering(r,kernel)
        mid_point_res  = merge_channels(mid_point_b, mid_point_g, mid_point_r)
        cv2.imwrite(filename + "_mid_point_filtered.jpg", np.array(mid_point_res))
