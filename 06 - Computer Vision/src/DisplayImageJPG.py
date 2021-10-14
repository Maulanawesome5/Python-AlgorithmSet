#Program display image, reference by Mr. Widodo 2016
import cv2
from PIL import Image
from pylab import *

#Baca image ke array
im = array(Image.open('D:/LATIHAN PEMROGRAMAN/PYTHON_ALGORITHM/06 - Computer Vision/src/image/yuki-yoda.jpg'))

#Plot image
imshow(im)

#Beberapa titik
x = [100, 100, 400, 400]
y = [200, 500, 200, 500]

#Plot titik dengan penanda garis merah
plot(x, y, 'r*')

#Line plot
plot(x[:2], y[:2])

#Tambahkan title
title('Plot : "Yuki-Yoda.jpg"')
show()