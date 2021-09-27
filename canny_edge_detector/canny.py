import cv2
import math
import numpy as np
import glob

class cannyEdgeDetector:
    def __init__(self, sigma=1, kernel_size=3, weak_pixel=75, strong_pixel=255, low_threshold=0.05,
                 high_threshold=0.15):
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        return

    def convolve(self, image, kernel):
        size = len(kernel)
        smallWidth = len(image) - (size - 1)
        smallHeight = len(image[0]) - (size - 1)
        output = [([0] * smallHeight) for i in range(smallWidth)]
        for i in range(smallWidth):
            for j in range(smallHeight):
                value = 0
                for n in range(size):
                    for m in range(size):
                        value = value + image[i + n][j + m] * kernel[n][m]
                output[i][j] = int(value)
        return output

    def gaussian_kernel(self, size, sigma):
        filter_size = int(size)
        gauss_kernel = [[0.0 for i in range(filter_size)] for i in range(filter_size)]
        m = filter_size // 2
        n = filter_size // 2
        for x in range(-m, m + 1):
            for y in range(-n, n + 1):
                x1 = 2.0 * math.pi * (sigma ** 2)
                x2 = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
                gauss_kernel[y + m][x + n] = (1 / x1) * x2
        return gauss_kernel

    def sobel_filters(self, img):
        Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        Ix = self.convolve(img, Kx)
        Iy = self.convolve(img, Ky)
        h, w = len(Ix), len(Ix[0])
        G = [[0 for i in range (w)] for i in range (h)]
        theta = [[0 for i in range (w)] for i in range (h)]
        for i in range(h):
            for j in range (w):
                G[i][j] = math.hypot(Ix[i][j], Iy[i][j])
                theta[i][j] = math.atan2(Ix[i][j], Iy[i][j])
        G_max = max(map(max, G))
        for i in range(h):
            for j in range(w):
                G[i][j] = G[i][j] / G_max * 255
        return (G, theta)

    def non_max_suppression(self, img, D):
        M, N = len(img), len(img[0])
        Z = [[0 for i in range (N)] for i in range (M)]
        for i in range(M):
            for j in range(N):
                D[i][j] = D[i][j] * 180. / math.pi
                if (D[i][j] < 0):
                    D[i][j] += 180
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255
                    # angle 0
                    if (0 <= D[i][j] < 22.5) or (157.5 <= D[i][j] <= 180):
                        q = img[i][j+1]
                        r = img[i][j-1]
                    # angle 45
                    elif (22.5 <= D[i][j] < 67.5):
                        q = img[i+1][j-1]
                        r = img[i-1][j+1]
                    # angle 90
                    elif (67.5 <= D[i][j] < 112.5):
                        q = img[i+1][j]
                        r = img[i-1][j]
                    # angle 135
                    elif (112.5 <= D[i][j] < 157.5):
                        q = img[i-1][j-1]
                        r = img[i+1][j+1]
                    if (img[i][j] >= q) and (img[i][j] >= r):
                        Z[i][j] = img[i][j]
                    else:
                        Z[i][j] = 0
                except IndexError as e:
                    pass
        return Z

    def threshold(self, img, lowthreshold=0.05, highthreshold=0.15, weak=75, strong=255):
        max_value = max(map(max, img))
        highThreshold = max_value * highthreshold
        lowThreshold = highThreshold * lowthreshold
        M, N = len(img), len(img[0])
        res = [[0 for k in range(N)] for i in range(M)]
        for i in range(len(res)):
            for k in range(len(res[0])):
                if img[i][k] >= highThreshold:
                    res[i][k] = strong
                elif (img[i][k] <= highThreshold) and (img[i][k] >= lowThreshold):
                    res[i][k] = weak
        return res

    def hysteresis(self, img, weak_pixel=75, strong_pixel=255):
        M, N = len(img), len(img[0])
        weak = int(weak_pixel)
        strong = int(strong_pixel)
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (img[i][j] == weak):
                    try:
                        if ((img[i + 1][j - 1] == strong) or (img[i + 1][j] == strong) or (img[i + 1][j + 1] == strong)
                                or (img[i][j - 1] == strong) or (img[i][j + 1] == strong)
                                or (img[i - 1][j - 1] == strong) or (img[i - 1][j] == strong) or (
                                        img[i - 1][j + 1] == strong)):
                            img[i][j] = strong
                        else:
                            img[i][j] = 0
                    except IndexError as e:
                        pass
        return img

    def detect(self, img):
        gauss_kernel = self.gaussian_kernel(self.kernel_size, self.sigma)
        img_smoothed = self.convolve(img, gauss_kernel)
        gradient_mat, theta_mat = self.sobel_filters(img_smoothed)
        non_max_img = self.non_max_suppression(gradient_mat, theta_mat)
        threshold_img = self.threshold(non_max_img, self.low_threshold, self.high_threshold, self.weak_pixel, self.strong_pixel)
        res = self.hysteresis(threshold_img)
        return res

if __name__ == "__main__":
    ext = ['jpg', 'jpeg', 'png']  # Add image formats here
    files = []
    [files.extend(glob.glob('*.' + e)) for e in ext]
    images = [cv2.imread(file, 0) for file in files]
    filenames = [file.split(".")[0] for file in files]
    detector = cannyEdgeDetector(sigma=1, kernel_size=3, low_threshold=0.05, high_threshold=0.15, weak_pixel=75, strong_pixel=255)
    for i in range(len(images)):
        img = images[i]
        filename = filenames[i]
        res = np.array(detector.detect(img))
        cv2.imwrite(filename + "_canny.png", res)

