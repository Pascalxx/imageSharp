import numpy as np
import cv2  # pip install -i https://pypi.douban.com/simple opencv-python==4.5.3.56


def image_sharp():
    img = cv2.imread('gData/lena_std.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dA = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1])
    # dA = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0])
    out_img = np.empty([img.shape[0], img.shape[1], 3])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            d = (dA[0] * img[i - 1][j - 1] if (i > 0 and j > 0) else 0) \
                + (dA[1] * img[i - 1][j] if (i > 0) else 0) \
                + (dA[2] * img[i - 1][j + 1] if (i > 0 and j < img.shape[1] - 1) else 0) \
                + (dA[3] * img[i][j - 1] if (j > 0) else 0) \
                + (dA[4] * img[i][j]) \
                + (dA[5] * img[i][j + 1] if (j < img.shape[1] - 1) else 0) \
                + (dA[6] * img[i + 1][j - 1] if (i < img.shape[0] - 1 and j > 0) else 0) \
                + (dA[7] * img[i + 1][j] if (i < img.shape[0] - 1) else 0) \
                + (dA[8] * img[i + 1][j + 1] if (i < img.shape[0] - 1 and j < img.shape[1] - 1) else 0)

            # d = cv2.add((dA[0] * img[i - 1][j - 1] if (i > 0 and j > 0) else 0), (dA[1] * img[i - 1][j] if (i > 0) else 0),
            #             dtype=cv2.CV_64F)
            # d = cv2.add(d.ravel(), dA[2] * img[i - 1][j + 1] if (i > 0 and j < img.shape[1] - 1) else 0, dtype=cv2.CV_64F)
            # d = cv2.add(d.ravel(), dA[3] * img[i][j - 1] if (j > 0) else 0, dtype=cv2.CV_64F)
            # d = cv2.add(d.ravel(), dA[4] * img[i][j], dtype=cv2.CV_64F)
            # d = cv2.add(d.ravel(), dA[5] * img[i][j + 1] if (j < img.shape[1] - 1) else 0, dtype=cv2.CV_64F)
            # d = cv2.add(d.ravel(), dA[6] * img[i + 1][j - 1] if (i < img.shape[0] - 1 and j > 0) else 0, dtype=cv2.CV_64F)
            # d = cv2.add(d.ravel(), dA[7] * img[i + 1][j] if (i < img.shape[0] - 1) else 0, dtype=cv2.CV_64F)
            # d = cv2.add(d.ravel(), dA[8] * img[i + 1][j + 1] if (i < img.shape[0] - 1 and j < img.shape[1] - 1) else 0,
            #             dtype=cv2.CV_64F)
            #
            # d = d.ravel()

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

    cv2.imshow('frame', out_img)
    while True:
        cv2.imshow('frame', out_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# # # # # # # # # # #
if __name__ == '__main__':
    image_sharp()
