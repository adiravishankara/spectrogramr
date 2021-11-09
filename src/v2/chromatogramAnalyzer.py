import cv2
import numpy as np
import matplotlib.pyplot as plt
from genChromatogram import genChromatogram as gc


class analyzer:
	def __init__(self, image):
		self.img = image

	def genChromatogram(self):
		self.gc = gc(self.img)
		self.gc.cvt2grey()
		self.gc.invert()
		self.chromatogram = self.gc.chromatogram()
		print(self.chromatogram.shape)

	def plot(self):
		plt.plot(range(self.chromatogram.shape[0]), self.chromatogram)
		plt.show()


def main():
	img = cv2.imread("test/img2.jpg")
	A = analyzer(img)
	A.genChromatogram()
	np.savetxt("PULSE.csv", A.chromatogram, fmt='%.10f', delimiter=',')
	cv2.imshow("1", A.img)
	cv2.waitKey(0)


if __name__ == "__main__":
	main()


