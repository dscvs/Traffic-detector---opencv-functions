import cv2
import numpy as np
import tkinter as tk
from tkinter import *
import pytesseract
from pprint import pprint
from PIL import Image, ImageTk

window = tk.Tk()
window.title("traffic")

img = cv2.imread("Resources/0.jpg")

def original_img():
    cv2.imshow('image',img)
    cv2.waitKey(0)

def inverted_img():
    img_i = (255-img)
    cv2.imshow('inverted', img_i)
    cv2.waitKey(0)

def grey_img():
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grey', img_g)
    cv2.waitKey(0)

def normal_img():
    img_n = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_n2 = cv2.equalizeHist(img_n)
    cv2.imshow('normalize histogram', img_n2)
    cv2.waitKey(0)

def scaleup_img():
    img_su = cv2.resize(img, None, fx=2, fy=2)
    cv2.imshow('scale up', img_su)
    cv2.waitKey(0)

def scaledown_img():
    img_sd = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow('scale down', img_sd)
    cv2.waitKey(0)

def gauss_img():
    img_g = cv2.GaussianBlur(img, (7, 7), 0)
    cv2.imshow('gauss', img_g)
    cv2.waitKey(0)

def sepia_img():
    img_s = cv2.transform(img, np.matrix([[0.272, 0.534, 0.131],[0.349, 0.686, 0.168],[0.393, 0.769, 0.189]]))
    cv2.imshow('sepia', img_s)
    cv2.waitKey(0)

def pencilsketch_img():
    img_p, _ = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    cv2.imshow('pencil sketch', img_p)
    cv2.waitKey(0)

def convert_img():
    img_c = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    cv2.imshow('convert', img_c)
    cv2.waitKey(0)

def rotate_img():
    img_r = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('rotate', img_r)
    cv2.waitKey(0)

def brightnesschange_img():
    value=30
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    img_output = cv2.merge((h, s, v))
    img_bc = cv2.cvtColor(img_output, cv2.COLOR_HSV2BGR)
    cv2.imshow('brightnesschange', img_bc)
    cv2.waitKey(0)

def edge_img():
    img_ed = cv2.Canny(img, 100, 300)
    cv2.imshow('edge detection', img_ed)
    cv2.waitKey(0)

def erosion_img():
    img_er = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(img_er, kernel, iterations=1)
    cv2.imshow('erosion', img_erosion)
    cv2.waitKey(0)

def dilation_img():
    img_d = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(img_d, kernel, iterations=1)
    cv2.imshow('dilation', img_dilation)
    cv2.waitKey(0)

def thresholding_img():
    retval, img_t = cv2.threshold(img, 125, 150, cv2.THRESH_BINARY)
    cv2.imshow('thresholding', img_t)

    grayT_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, img_t2 = cv2.threshold(grayT_img, 10, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresholding2', img_t2)

    img_t3 = cv2.adaptiveThreshold(grayT_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    cv2.imshow('thresholding3 - adaptive', img_t3)

    retval2, img_t4 = cv2.threshold(grayT_img, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('thresholding4 - otsu', img_t4)
    cv2.waitKey(0)

def skeletonization_img():
    img_s = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_s = cv2.threshold(img_s, 127, 255, 0)
    skel = np.zeros(img_s.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        open_s = cv2.morphologyEx(img_s, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img_s, open_s)
        eroded = cv2.erode(img_s, element)
        skel = cv2.bitwise_or(skel, temp)
        img_s = eroded.copy()

        if cv2.countNonZero(img_s) == 0:
            break
            
    cv2.imshow('skeletonization', skel)
    cv2.waitKey(0)

def segmentation_img():
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('segmentation - thresh', thresh)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    cv2.imshow('segmentation - dist_transform', dist_transform)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    cv2.imshow('segmentation - sure_bg', sure_bg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.imshow('segmentation - sure_fg', sure_fg)


# def ocr_img():
#     img = cv2.imread("traffic.png")
#     pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
#
#     gray_ocr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     ret, thresh1 = cv2.threshold(gray_ocr, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
#     rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
#     dilation_ocr = cv2.dilate(thresh1, rect_kernel, iterations=1)
#
#     contours, hierarchy = cv2.findContours(dilation_ocr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#     im2 = img.copy()
#
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         cropped = im2[y:y + h, x:x + w]
#         text = pytesseract.image_to_string(cropped)
#         print(text)


#PROCES DETECTION
def procesdetection_img():
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # blue green red

    #Red Color range
    low_red = np.array([160,100,100])
    hi_red = np.array([180,255,255])
    red_mask = cv2.inRange(hsv_img, low_red, hi_red)
    red = cv2.bitwise_and(img, img, mask=red_mask)

    #Orange Color range
    low_orange = np.array([15,150,150])
    hi_orange = np.array([35,255,255])
    orange_mask = cv2.inRange(hsv_img, low_orange, hi_orange)
    orange = cv2.bitwise_and(img, img, mask=orange_mask)

    #Green Color range
    low_green = np.array([40,50,50])
    hi_green = np.array([90,255,255])
    green_mask = cv2.inRange(hsv_img, low_green, hi_green)
    green = cv2.bitwise_and(img, img, mask=green_mask)

    #detection RED
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(img, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)
            cv2.putText(imageFrame, "Red", (x, y), #text structure
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, #text size
                        (0, 0, 255)) #text color

    #detection GREEN
    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(img, (x, y),
                                       (x + w, y + h),
                                       (0, 255, 0), 2)

            cv2.putText(imageFrame, "Green", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0))

    #detection orange
    contours, hierarchy = cv2.findContours(orange_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(img, (x, y),
                                       (x + w, y + h),
                                       (0, 165, 255), 2)

            cv2.putText(imageFrame, "Orange", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 165, 255))

    cv2.imshow("Traffic", img)




labelframe = LabelFrame(window)
labelframe.pack(fill="both", expand="yes")
left = Label(labelframe)

original = tk.Button(window, text="Oryginalne zdjęcie", width=30, command=original_img)
original.pack()

procesdetection = tk.Button(window, text="Detekcja koloru sygnalizacji świetlnej", width=30, command=procesdetection_img)
procesdetection.pack()

inverted = tk.Button(window, text="Negatyw", width=30, command=inverted_img)
inverted.pack()

grey = tk.Button(window, text="Konwersja do odcieni szarości", width=30, command=grey_img)
grey.pack()

normal = tk.Button(window, text="Normalizacja histogramu", width=30, command=normal_img)
normal.pack()

scaleup = tk.Button(window, text="Skalowanie (większe)", width=30, command=scaleup_img)
scaleup.pack()

scaledown = tk.Button(window, text="Skalowanie (mniejsze)", width=30, command=scaledown_img)
scaledown.pack()

gauss = tk.Button(window, text="Rozmycie Gaussa", width=30, command=gauss_img)
gauss.pack()

sepia = tk.Button(window, text="Sepia", width=30, command=sepia_img)
sepia.pack()

pencilsketch = tk.Button(window, text="Szkic ołówkiem", width=30, command=pencilsketch_img)
pencilsketch.pack()

convert = tk.Button(window, text="Transformacja przestrzeni barw", width=30, command=convert_img)
convert.pack()

rotate = tk.Button(window, text="Obrót", width=30, command=rotate_img)
rotate.pack()

brightnesschange = tk.Button(window, text="Zmiana jasności", width=30, command=brightnesschange_img)
brightnesschange.pack()

edgedetection = tk.Button(window, text="Detekcja krawędzi", width=30, command=edge_img)
edgedetection.pack()

erosion = tk.Button(window, text="Erozja", width=30, command=erosion_img)
erosion.pack()

dilation = tk.Button(window, text="Dylatacja", width=30, command=dilation_img)
dilation.pack()

thresholding = tk.Button(window, text="Progowanie", width=30, command=thresholding_img)
thresholding.pack()

skeletonization = tk.Button(window, text="Szkieletyzacja", width=30, command=skeletonization_img)
skeletonization.pack()

segmentation = tk.Button(window, text="Segmentacja", width=30, command=segmentation_img)
segmentation.pack()


tk.mainloop()