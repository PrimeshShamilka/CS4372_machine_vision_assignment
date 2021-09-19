import cv2
import glob
import numpy as np
       
def mean_filtering(image,kernel):
    image = wrap(image)
    kernel = get_mean_filter(kernel)
    size = len(kernel)
    smallWidth = len(image) - (size-1)
    smallHeight = len(image[0]) - (size-1)
    output = [([0]*smallHeight) for i in range(smallWidth)]
    for i in range(smallWidth):
        for j in range (smallHeight):
            value = 0
            for n in range(size):
                for m in range(size):
                    value = value + image[i+n][j+m]*kernel[n][m]
            output[i][j] = int(value)      
    return output 
def medain_filtering(image,kernel):
    image = wrap(image)
    size = len(kernel)
    smallWidth = len(image) - (size-1)
    smallHeight = len(image[0]) - (size-1)
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
    size = len(kernel)
    smallWidth = len(image) - (size-1)
    smallHeight = len(image[0]) - (size-1)
    output = [([0]*smallHeight) for i in range(smallWidth)]
    for i in range(smallWidth):
        for j in range (smallHeight):
            array = []
            for n in range(size):
                for m in range(size):
                    array.append(image[i+n][j+m])       
            mid = (int(min(array)) + int(max(array)))//2
            output[i][j] = int(mid)
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

def mergeChannels(b,g,r):
    image = []
    for i in range(len(r)):
        row = []
        for j in range(len(r[0])):
            row.append([b[i][j],g[i][j],r[i][j]])  
        image.append(row)
    return image 
if __name__ == "__main__":
    kernel = [[1,1,1],[1,1,1],[1,1,1]]
    ext = ['jpg','jpeg']
    files = []
    [files.extend(glob.glob('*.' + e)) for e in ext]
    images = [cv2.imread(file) for file in files]
    filenames = [file.split(".")[0] for file in files]
    for i in range(len(images)):
        img = images[i]
        filename = filenames[i]
        #mean_filtering
        b = img[:,:,0]
        g = img[:,:,1]
        r = img[:,:,2]
        mean_out_b = np.array(mean_filtering(b,kernel))
        mean_out_g = np.array(mean_filtering(g,kernel))
        mean_out_r = np.array(mean_filtering(r,kernel))
        mean_out  = np.array(mergeChannels(mean_out_b,mean_out_g,mean_out_r))
        cv2.imwrite(filename + "_mean_filtered.jpg",mean_out)
        #medain_filtering
        medain_out_b = medain_filtering(b,kernel)
        medain_out_g = medain_filtering(g,kernel)
        medain_out_r = medain_filtering(r,kernel)
        medain_out  = np.array(mergeChannels(medain_out_b,medain_out_g,medain_out_r))
        cv2.imwrite(filename + "_median_filtered.jpg",medain_out)

        #midpoint_filtering
        mid_point_out_b = mid_point_filtering(b,kernel)
        mid_point_out_g = mid_point_filtering(g,kernel)
        mid_point_out_r = mid_point_filtering(r,kernel)
        mid_point_out  = np.array(mergeChannels(mid_point_out_b,mid_point_out_g,mid_point_out_r))
        cv2.imwrite(filename + "_mid_point_filtered.jpg",mid_point_out)



