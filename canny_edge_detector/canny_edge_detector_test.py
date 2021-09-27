from scipy import ndimage
# from scipy.ndimage.filters import convolve

# from scipy import misc
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

class cannyEdgeDetector:
    def __init__(self, imgs, sigma=1, kernel_size=3, weak_pixel=75, strong_pixel=255, lowthreshold=0.05,
                 highthreshold=0.15):
        self.imgs = imgs
        self.imgs_final = []
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        return

    def convolution(self, kernel, image, edge_param=""):
        i_height = len(image)
        i_width = len(image[0])
        k_height = len(kernel)
        k_width = len(kernel[0])
        filtered = [[0 for i in range(i_width)] for i in range (i_height)]
        for y in range(i_height):
            for x in range(i_width):
                # Iterate over the kernel
                weighted_pixel_sum = 0
                m = k_height // 2
                n = k_width // 2
                for ky in range(-m, m + 1):
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

    def gaussian_kernel(self, size, sigma=1):
        # filter_size = int(size)
        # # filter_size = 2 * int(4 * sigma + 0.5) + 1
        # # gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
        # gauss_kernel = [[0.0 for i in range(filter_size)] for i in range(filter_size)]
        # m = filter_size // 2
        # n = filter_size // 2
        # for x in range(-m, m + 1):
        #     for y in range(-n, n + 1):
        #         x1 = 2.0 * math.pi * (sigma ** 2)
        #         x2 = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        #         gauss_kernel[y + m][x + n] = (1 / x1) * x2
        #         # gaussian_filter[x+m, y+n] = (1/x1)*x2
        # return gauss_kernel
        x = [[-1 + k for i in range(size)] for k in range(size)]
        y = [[-1, 0, 1] for k in range(size)]
        g = [[0, 0, 0] for k in range(size)]
        normal = 1 / (2.0 * math.pi * sigma ** 2)
        for i in range(size):
            for k in range(size):
                g[i][k] = math.exp(-((x[i][k] ** 2 + y[i][k] ** 2) / (2.0 * sigma ** 2))) * normal
        return g

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

    def sobel_filters(self, img):
        # # Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        # # Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        # Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        # Ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        # # Ix = ndimage.filters.convolve(img, Kx)
        # # Iy = ndimage.filters.convolve(img, Ky)
        # Ix = self.convolution(Kx, img)
        # Iy = self.convolution(Ky, img)
        # h, w = len(img), len(img[0])
        # G = [[0 for i in range (w)] for i in range (h)]
        # theta = [[0 for i in range (w)] for i in range (h)]
        # for i in range(h):
        #     for j in range (w):
        #         G[i][j] = math.hypot(Ix[i][j], Iy[i][j])
        #         theta[i][j] = math.atan2(Ix[i][j], Iy[i][j])
        # G_max = max(map(max, G))
        # for i in range(h):
        #     for j in range(w):
        #         G[i][j] = G[i][j] / G_max * 255
        # # G = np.hypot(Ix, Iy)
        # # G = G / G.max() * 255
        # # theta = np.arctan2(Iy, Ix)
        # return (G, theta)

        Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

        # Ix = self.convolution(Kx, img)
        # Iy = self.convolution(Ky, img)

        Ix = self.convolve(img, Kx)
        Iy = self.convolve(img, Ky)

        G = [[0 for k in range(len(Ix[0]))] for i in range(len(Ix))]
        theta = [[0 for k in range(len(Ix[0]))] for i in range(len(Ix))]
        for i in range(len(Ix)):
            for k in range(len(Ix[0])):
                G[i][k] = math.hypot(Ix[i][k], Iy[i][k])
                theta[i][k] = math.atan2(Ix[i][k], Iy[i][k])
        g_max = max(map(max, G))
        G = [[G[i][k] / g_max * 255 for k in range(len(Ix[0]))] for i in range(len(Ix))]
        return (G, theta)

    def non_max_suppression(self, img, D):
        M, N = len(img), len(img[0])
        # Z = np.zeros((M, N), dtype=np.int32)
        Z = [[0 for i in range (N)] for i in range (M)]
        # angle = [[0 for i in range(N)] for i in range(M)]
        for i in range(M):
            for j in range(N):
                D[i][j] = D[i][j] * 180. / math.pi
                if (D[i][j] < 0):
                    D[i][j] += 180
                # angle[i][j] = val
        # angle = D * 180. / math.pi
        # angle[angle < 0] += 180
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
                    if (D[i][j] >= q) and (D[i][j] >= r):
                        Z[i][j] = D[i][j]
                    else:
                        Z[i][j] = 0
                except IndexError as e:
                    pass
        return Z

    def threshold(img, lowthreshold=0.05, highthreshold=0.15, weak=75, strong=255):
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

    def hysteresis(self, img):
        M, N = len(img), len(img[0])
        weak = int(self.weak_pixel)
        strong = int(self.strong_pixel)
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

    def detect(self):
        imgs_final = []
        for i, img in enumerate(self.imgs):
            # self.img_smoothed = self.convolution(self.gaussian_kernel(self.kernel_size, sigma=self.sigma), img)
            self.img_smoothed = self.convolve(img, self.gaussian_kernel(self.kernel_size, sigma=self.sigma))
            self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
            nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
            thresholdImg = self.threshold(nonMaxImg)
            img_final = self.hysteresis(thresholdImg)
            self.imgs_final.append(img_final)

        return img_final


if __name__ == "__main__":
    img = cv2.imread("/media/primesh/F4D0EA80D0EA4906/ACADEMICS/MORA/Sem 7/Machine vision/canny_edge_detector/canny_edge_detector/faces_imgs/lena.png",0)
    img_mat = []
    for i in range(0, img.shape[0]):
        row = []
        for j in range(0, img.shape[1]):
            pixel = img.item(i, j)
            row.append(pixel)
        img_mat.append(row)

    detector = cannyEdgeDetector([img_mat], sigma=1, kernel_size=3, lowthreshold=0.05, highthreshold=0.15, weak_pixel=75, strong_pixel=255)
    imgs_final = detector.detect()
    imgs_final = np.array(imgs_final)
    cv2.imwrite('lena_canny.png', imgs_final)