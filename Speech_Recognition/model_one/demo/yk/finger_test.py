import numpy as np
import cv2

skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
minValue = 70


def skinMask(im):
    # HSV values
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)
    mask = cv2.erode(mask, skinkernel, iterations=1)
    mask = cv2.dilate(mask, skinkernel, iterations=1)

    # blur
    mask = cv2.GaussianBlur(mask, (15, 15), 1)

    # bitwise and mask original frame
    res = cv2.bitwise_and(im, im, mask=mask)

    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    return res


def binaryMask(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return res


def img_capture():
    cap = cv2.VideoCapture(-1)
    while 1:
        _, origimg = cap.read()
        cv2.imshow('im', origimg)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite('t.png', origimg)
        elif key == ord('q') or key == 27 or key == 32:
            break
    cap.release()


def LocFinger(img, roi):
    mask = skinMask(img)
    center = ((roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2)
    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.pointPolygonTest(contour, center, 0) > 0]
    if not len(contours):
        return [], None, None

    # find the contour with max length
    contour_len = [len(contour) for contour in contours]
    contour_len = np.array(contour_len)
    max_length_index = contour_len.argmax()
    contour = contours[max_length_index]
    contour = np.squeeze(contour)

    # contour[0]: the point with max height
    finger_loc = contour[0][0], contour[0][1]

    # find bbox of contour
    max_loc = contour.max(0)
    min_loc = contour.min(0)
    # x_mid = (min_loc[0] + max_loc[0]) / 2
    # y_mid = (min_loc[1] + max_loc[1]) / 2
    # delta = 224
    roi_width = roi[2] - roi[0]
    if min_loc[0] < roi[0] - roi_width * 0.2:
        min_loc[0] = roi[0] - roi_width * 0.2
    if max_loc[0] > roi[2] + roi_width * 0.2:
        max_loc[0] = roi[2] + roi_width * 0.2
    if max_loc[1] > roi[3] + (roi[3] - roi[1]) * 0.2:
            max_loc[1] = roi[3] + (roi[3] - roi[1]) * 0.2
    bbox_contour = (min_loc[0], min_loc[1]), (max_loc[0], max_loc[1])

    return [contour], finger_loc, bbox_contour


if __name__ == '__main__':

    im = cv2.imread('/home/yinkang/0MyProject/0_HandSSD/test.png')
    mask = skinMask(im)
    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    print contours[0]

    """
    im = cv2.imread('/home/yinkang/0MyProject/0_HandSSD/test.png')
    bbox = [404, 108, 566, 406]
    contour = LocFinger(im, bbox)
    cv2.drawContours(im, contour, -1, (0, 0, 255), 3)
    cv2.circle(im, (contour[0][0][0], contour[0][0][1]), 5, (0, 255, 0), 3)
    cv2.imshow('im', im)
    cv2.waitKey()
    """

    # img_capture()
    # exit()
    # im = cv2.imread('/home/yinkang/0MyProject/0_HandSSD/test.png')
    # mask = skinMask(im)
    # bbox = [404, 108, 566, 406]
    # cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))
    #
    # mask[mask != 0] = 255
    # # cv2.imshow('mask', mask)
    #
    # contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    # # contours = [cv2.approxPolyDP(cnt, 5, True) for cnt in contours]
    # # print len(hierarchy), hierarchy
    # # print len(contours), contours
    # # print contours[0]
    #
    # # print len(contours)
    # contour = contours[3]
    # print type(contour)
    # contour = contour[:, :, -1]
    # print contour.min()
    #
    # # cv2.drawContours(im, contour, -1, (0, 0, 255), 3)
    # cv2.imshow('im', im)
    # cv2.waitKey()
