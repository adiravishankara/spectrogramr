import PyQt5.QtWidgets
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

from PyQt5 import *


class genChromatogram:
	def __init__(self, image):
		self.img = image
		print(self.img.shape)
		self.dimensions = self.img.shape
		#rint("Image Loaded")

	def crop(self):
		self.img = self.img[10:100, 10:100]  # xmin:xmax, ymin:ymax

	def cvt2grey(self):
		self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

	def flip(self):
		self.img = cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)
		self.dimensions = self.img.shape

	def invert(self):
		self.img = cv2.bitwise_not(self.img)

	def chromatogram(self):
		self.flatten = np.zeros(self.dimensions[1])
		for i in range(self.dimensions[1]):
			for j in range(self.dimensions[0]):
				self.flatten[i] = self.flatten[i] + self.img[j, i]

		return self.flatten

	def plot(self):
		plt.plot(self.flatten)
		plt.show()


def main():
	app = PyQt5.QtWidgets.QApplication(sys.argv)
	file, somethign= PyQt5.QtWidgets.QFileDialog.getOpenFileName(caption="Load File")
	print(file)
	img = cv2.imread(file)
	a = genChromatogram(img)
	a.invert()
	a.cvt2grey()
	a.chromatogram()
	b = a.flatten
	print(b.shape)
	newFile, ok = PyQt5.QtWidgets.QFileDialog.getSaveFileName(caption="Save File")
	plt.plot(range(len(b)), b)
	plt.show()
	np.savetxt("{}.csv".format(newFile), b, fmt='%.10f', delimiter=',')






if __name__ == "__main__":
	main()
