import matplotlib.pyplot as plt
import numpy as np
import cv2

def diffAB(fileA, fileB, fileResult):
    imgA = cv2.imread(fileA)
    imgB = cv2.imread(fileB)

    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    hight, width = grayA.shape

    result_window = np.zeros((hight, width), dtype=imgA.dtype)
    for start_y in range(0, hight-100, 50):
        for start_x in range(0, width-100, 50):
            window = grayA[start_y:start_y+100, start_x:start_x+100]
            match = cv2.matchTemplate(grayB, window, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(match)
            matched_window = grayB[max_loc[1]:max_loc[1]+100, max_loc[0]:max_loc[0]+100]
            result = cv2.absdiff(window, matched_window)
            result_window[start_y:start_y+100, start_x:start_x+100] = result

    result_window_bin = cv2.threshold(result_window, 127, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(result_window_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgC = imgA.copy()
    for contour in contours:
        min = np.nanmin(contour, 0)
        max = np.nanmax(contour, 0)
        loc1 = (min[0][0], min[0][1])
        loc2 = (min[0][0], min[0][1])
        cv2.rectangle(imgC, loc1, loc2, 255, 2)

    cv2.imwrite(fileResult, imgC)

diffAB('./fileA.jpg', './fileB.jpg', 'diffResult.jpg')
