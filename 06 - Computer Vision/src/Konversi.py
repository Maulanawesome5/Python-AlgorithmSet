#Program konversi image, reference by Mr. Widodo 2016
import cv2
from PIL import Image
from pylab import *
import os

filelist = array(Image.open('D:/LATIHAN PEMROGRAMAN/PYTHON_ALGORITHM/06 - Computer Vision/src/image/yuki-yoda.jpg'))

for infile in filelist:
    outfile = os.path.splitext(infile) [0] + ".PNG"
    if infile != outfile:
        try:
            Image.open(infile).save(outfile)
        except IOError:
            print("Cannot convert", infile)

grayImage = cv2.imread('MyPic.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
cv2.imwrite('MyPicGray.png', grayImage)
