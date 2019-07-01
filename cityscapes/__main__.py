from skunkwork.utils import getListOfFiles, clog
import cv2
import numpy as np
import time

def resizeImage(img, scale_percent=20):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def grayImage(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def showImage(img, name='window'):
    cv2.imshow(name, img)


def main():
    path_ = '/media/sameera/BASS/'
    path_data = path_ + 'data'
    path_labels = path_ + 'labels'

    data = getListOfFiles(path_data, sort=True)
    labels = getListOfFiles(path_labels, sort=True)
    # fitering specific label
    labels = [x for x in labels if 'instanceIds' in x]

    clog(len(data), len(labels))

    # read images by cv2
    for n in range(7, 10):
        clog(data[n], labels[n])

        # data
        img = cv2.imread(data[n])
        img = resizeImage(img)
        showImage(img, data[n])

        # label
        img = cv2.imread(labels[n])
        img = resizeImage(img)
        img = grayImage(img)
        # showImage(img, 'before')

        st = time.time()
        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
        _, img = cv2.threshold(img, 102, 255, cv2.THRESH_TOZERO_INV)
        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        clog(time.time()-st)

        showImage(img, labels[n])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
