#Program contour image

from PIL import Image
from pylab import *

#Baca image ke array dan konversi ke grayscale
im = array(Image.open('D:/LATIHAN PEMROGRAMAN/PYTHON_ALGORITHM/06 - Computer Vision/src/image/yuki-yoda.jpg').convert('L'))

#Buat gambar baru
figure()

#Ubah menjadi gray
gray()

#Tampilkan contour
contour(im, origin='image')
axis('equal')
axis('off')
figure()
hist(im.flatten(), 128)
show()