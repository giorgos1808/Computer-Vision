import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def myColorToGray(A, gamma=1, weight_input=False):
    R, G, B = A[:, :, 2], A[:, :, 1], A[:, :, 0]

    if not weight_input:
        r_const, g_const, b_const = 0.25, 0.7, 0.05
    elif weight_input:
        while True:
            r_const = float(input('Give weight of Red: '))
            b_const = float(input('Give weight of Blue: '))
            g_const = float(input('Give weight of Green: '))

            if r_const + g_const + b_const != 1:
                print('Wrong weights')
            else:
                break

    gray_img = r_const*R**gamma + g_const*G**gamma + b_const*B**gamma

    return gray_img


def myConv2(A, B, pad=[0,0], stride=[1, 1], is_equal = False):
    # Create numpy arrays
    A = np.array(A)
    B = np.array(B)

    # Fetch the dimensions for iteration over the pixels and weights
    i_height, i_width = A.shape[0], A.shape[1]
    k_height, k_width = B.shape[0], B.shape[1]

    if type(pad) == list:
        pad_height, pad_width = pad[0], pad[1]
    elif type(pad) == int:
        pad_height, pad_width = pad, pad

    if type(stride) == list:
        stride_height, stride_width = stride[0], stride[1]
    elif type(stride) == int:
        stride_height, stride_width = stride, stride

    if is_equal:
        p = 0
        s = 1
        while p == 0:
            pad_height = ( (i_height - 1) * s - i_height + k_height ) / 2
            pad_width = ( (i_width - 1) * s - i_width + k_width ) / 2

            if pad_width.is_integer() and pad_height.is_integer():
                p = 1
                pad_height = int(pad_height)
                pad_width = int(pad_width)
                stride_height = s
                stride_width = s
            else:
                s += 1

    if ((i_height - k_height + pad_height * 2 + stride_height) / stride_height).is_integer() and ((i_width - k_width + pad_width * 2 + stride_width) / stride_width).is_integer():
        padding_image = np.zeros((i_height + pad_height * 2, i_width + pad_width * 2))

        if pad_width > 0 and pad_height == 0:
            padding_image[:, pad_width:-pad_width] = A[:, :]
        elif pad_width == 0 and pad_height > 0:
            padding_image[pad_height:-pad_height, :] = A[:, :]
        elif pad_width > 0 and pad_height > 0:
            padding_image[pad_width:-pad_width, pad_height:-pad_height] = A[:, :]
        elif pad_width == 0 and pad_height == 0:
            padding_image = A

        feature_map = np.zeros((int((i_height - k_height + pad_height * 2 + stride_height) / stride_height),int((i_width - k_width + pad_width * 2 + stride_width) / stride_width)))
        j = 0
        j_fm = 0
        while (j + k_height) <= i_height + pad_height * 2:
            i = 0
            i_fm = 0
            while (i + k_width) <= i_width + pad_width * 2:
                feature_map[j_fm:k_height + j_fm, i_fm:k_width + i_fm] = (np.sum(padding_image[j:k_height + j, i:k_width + i] * B))
                i += stride_width
                i_fm += 1

            if (i + k_width) > i_width + pad_width * 2:
                j += stride_height
                j_fm += 1

        return feature_map
    else:
        return 0


def myImFilter(A, kernel=(3,3), method='mean'):
    i_height, i_width = A.shape[0], A.shape[1]
    k_height, k_width = kernel[0], kernel[1]

    filter = np.zeros([k_height, k_width])

    if method == 'mean':
        filter[:, :] = 1 / filter.size
        filtered_img = myConv2(A, filter, is_equal=True)
    elif method == 'median':
        padding_image = np.zeros([i_height + 2, i_width + 2])
        padding_image[1:-1, 1:-1] = A[:, :]

        filtered_img = np.zeros([i_height - k_height + 2 + 1, i_width - k_width + 2 + 1])
        j = 0
        while (j + k_height) <= i_height + 2:
            i = 0
            while (i + k_width) <= i_width + 2:
                vec = padding_image[j:k_height + j, i:k_width + i]
                sorted_vec = np.sort(vec.reshape(1, vec.size))
                filtered_img[j:k_height + j, i:k_width + i] = sorted_vec[0, int((sorted_vec.size+1)/2 - 1)]
                i += 1

            if (i + k_width) > i_width + 2:
                j += 1

    return filtered_img


def myImNoise(A, noise='gauss'):
    A = A / 255
    i_height, i_width = A.shape[0], A.shape[1]

    if noise == 'gauss':
        # Gaussian Noise
        mean = 0
        var = 0.01
        sigma = np.sqrt(var)

        noise = np.random.normal(loc=mean, scale=sigma, size=(i_height, i_width))
        img_noise = A + noise

    elif noise == 'salt_and_pepper':
        # Salt and Pepper

        # Salt and Pepper Amount
        pepper = 0.05
        salt = 1 - pepper

        img_noise = np.zeros((i_height, i_width))
        for h in range(i_height):
            for w in range(i_width):
                random_noise = np.random.random()
                if random_noise < pepper:
                    img_noise[h][w] = 0
                elif random_noise > salt:
                    img_noise[h][w] = 1
                else:
                    img_noise[h][w] = A[h][w]

    return img_noise


def myEdgeDetection(A, method='sobel'):
    A = np.array(A)

    if method == 'sobel':
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        img_x = myConv2(A, kernel_x, is_equal=True)
        img_y = myConv2(A, kernel_y, is_equal=True)
        output_img = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))

    elif method == 'prewitt':
        kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        img_x = myConv2(A, kernel_x, is_equal=True)
        img_y = myConv2(A, kernel_y, is_equal=True)
        output_img = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))

    elif method == 'laplace':
        kernel = np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]])
        image = myConv2(A, kernel, is_equal=True)

        # Zero Cross Detection
        output_img = np.zeros(image.shape)
        for i in range(0, image.shape[0] - 1):
            for j in range(0, image.shape[1] - 1):
                if image[i][j] > 0:
                    if image[i + 1][j] < 0 or image[i + 1][j + 1] < 0 or image[i][j + 1] < 0:
                        output_img[i, j] = 1
                elif image[i][j] < 0:
                    if image[i + 1][j] > 0 or image[i + 1][j + 1] > 0 or image[i][j + 1] > 0:
                        output_img[i, j] = 1

    return output_img


if __name__ == '__main__':

    image = cv.imread('gfg.png')
    image_rgb = image[:, :, ::-1]
    plt.figure('Original Image')
    plt.imshow(image_rgb)
    plt.show()

    A = myColorToGray(image)
    plt.figure('A')
    plt.imshow(A, cmap=plt.cm.get_cmap('gray'))
    plt.show()

    B = myImNoise(A, noise='gauss')
    plt.figure('B')
    plt.imshow(B, cmap=plt.cm.get_cmap('gray'))
    plt.show()

    C = myImFilter(B, kernel=(3,3), method='mean')
    plt.figure('C')
    plt.imshow(C, cmap=plt.cm.get_cmap('gray'))
    plt.show()

    A_edge = myEdgeDetection(A, method='sobel')
    plt.figure('A_edge')
    plt.imshow(A_edge, cmap=plt.cm.get_cmap('gray'))
    plt.show()

    Method = ['sobel', 'prewitt', 'laplace']
    for i in range(3):
        E = myImFilter(A, kernel=(9,9), method='mean')
        F = myImFilter(A, kernel=(3,3), method='mean')

        E_edge = myEdgeDetection(A, method=Method[i])
        plt.figure('E_edge '+Method[i])
        plt.imshow(E_edge, cmap=plt.cm.get_cmap('gray'))
        plt.show()

        F_edge = myEdgeDetection(A, method=Method[i])
        plt.figure('F_edge '+Method[i])
        plt.imshow(F_edge, cmap=plt.cm.get_cmap('gray'))
        plt.show()
