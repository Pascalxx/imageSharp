import numpy as np
import cv2


# Laplacian
def do_laplacian(img_input):
    print('do_laplacian is running...')
    dA = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1])
    # dA = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0])
    out_img = np.empty([img_input.shape[0], img_input.shape[1]])

    # 逐一處理像素
    for i in range(1, img_input.shape[0] - 1):
        for j in range(1, img_input.shape[1] - 1):
            d = (dA[0] * img_input[i - 1][j - 1] if (i > 0 and j > 0) else 0) \
                + (dA[1] * img_input[i - 1][j] if (i > 0) else 0) \
                + (dA[2] * img_input[i - 1][j + 1] if (i > 0 and j < img_input.shape[1] - 1) else 0) \
                + (dA[3] * img_input[i][j - 1] if (j > 0) else 0) \
                + (dA[4] * img_input[i][j]) \
                + (dA[5] * img_input[i][j + 1] if (j < img_input.shape[1] - 1) else 0) \
                + (dA[6] * img_input[i + 1][j - 1] if (i < img_input.shape[0] - 1 and j > 0) else 0) \
                + (dA[7] * img_input[i + 1][j] if (i < img_input.shape[0] - 1) else 0) \
                + (dA[8] * img_input[i + 1][j + 1] if (
                    i < img_input.shape[0] - 1 and j < img_input.shape[1] - 1) else 0)

            # d[0] = 255 if d[0] > 255 else d[0]
            # d[0] = 0 if d[0] < 0 else d[0]
            # d[1] = 255 if d[1] > 255 else d[1]
            # d[1] = 0 if d[1] < 0 else d[1]
            # d[2] = 255 if d[2] > 255 else d[2]
            # d[2] = 0 if d[2] < 0 else d[2]
            d = d / 255
            d = 255 if d > 255 else d
            d = 0 if d < 0 else d

            out_img[i][j] = d

    print('do_laplacian done!')
    # cv2.imshow('frame', out_img)

    return out_img


# 一階微分處理
def do_first_order(img_input):
    print('do_first_order is running...')
    out_img = np.empty([img_input.shape[0], img_input.shape[1]])

    # 逐一處理像素
    for i in range(1, img_input.shape[0] - 1):
        for j in range(1, img_input.shape[1] - 1):
            d1 = img_input[i + 1][j - 1] + 2 * img_input[i + 1][j] + img_input[i + 1][j + 1]
            d2 = img_input[i - 1][j - 1] + 2 * img_input[i - 1][j] + img_input[i - 1][j + 1]
            d3 = img_input[i - 1][j + 1] + 2 * img_input[i][j + 1] + img_input[i + 1][j + 1]
            d4 = img_input[i - 1][j - 1] + 2 * img_input[i][j - 1] + img_input[i + 1][j - 1]
            d = abs(d1 - d2) + abs(d3 - d4)

            d = d / 255
            d = 255 if d > 255 else d
            d = 0 if d < 0 else d
            out_img[i][j] = d

    print('do_first_order done!')
    # cv2.imshow('frame', out_img)

    return out_img


# 模糊處理
def do_median_filter(img_input):
    print('do_median_filter is running...')
    out_img = np.empty([img_input.shape[0], img_input.shape[1]])

    # 逐一處理像素
    for i in range(1, img_input.shape[0] - 1):
        for j in range(1, img_input.shape[1] - 1):
            d = (img_input[i - 1][j - 1] if (i > 0 and j > 0) else 0) \
                + (img_input[i - 1][j] if (i > 0) else 0) \
                + (img_input[i - 1][j + 1] if (i > 0 and j < img_input.shape[1] - 1) else 0) \
                + (img_input[i][j - 1] if (j > 0) else 0) \
                + (img_input[i][j]) \
                + (img_input[i][j + 1] if (j < img_input.shape[1] - 1) else 0) \
                + (img_input[i + 1][j - 1] if (i < img_input.shape[0] - 1 and j > 0) else 0) \
                + (img_input[i + 1][j] if (i < img_input.shape[0] - 1) else 0) \
                + (img_input[i + 1][j + 1] if (
                    i < img_input.shape[0] - 1 and j < img_input.shape[1] - 1) else 0)

            d = 2 * d / 9
            # d = d / 255
            d = 255 if d > 255 else d
            d = 0 if d < 0 else d
            out_img[i][j] = d

    print('do_median_filter done!')
    # cv2.imshow('frame', out_img)

    return out_img


# 一階與二階影像相乘
def do_multiple(img_input_fst, img_input_sec):
    print('do_multiple is running...')
    out_img = np.empty([img_input_fst.shape[0], img_input_fst.shape[1]])

    # 逐一處理像素
    for i in range(out_img.shape[0]):
        for j in range(out_img.shape[1]):
            out_img[i][j] = img_input_fst[i][j] * img_input_sec[i][j]

    print('do_multiple done!')
    return out_img


# 原始影像銳化
def do_image_sharp(img_input, multiple_img_input):
    print('do_image_sharp is running...')
    out_img = np.empty([img_input.shape[0], img_input.shape[1]])
    for i in range(img_input.shape[0]):
        for j in range(img_input.shape[1]):
            out_img[i][j] = img_input[i][j] + multiple_img_input[i][j]

    print('do_image_sharp done!')
    cv2.imshow('frame_end', out_img)


# # # # # # # # # # #
if __name__ == '__main__':
    img = cv2.imread('gData/lena_std.jpg')
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉灰階
    laplacian_img = do_laplacian(img_bw)  # 進行Laplacian處理
    first_order_img = do_first_order(img_bw)  # 進行一階微分處理
    median_filter_img = do_median_filter(first_order_img)  # 模糊去雜訊
    multiple_img = do_multiple(first_order_img, laplacian_img)  # 一階與二階影像相乘

    cv2.imshow('frame_o', img_bw)
    do_image_sharp(img_bw, multiple_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
