import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


class genSpectrogram:
	def __init__(self, *args, **kwargs):
		super(genSpectrogram, self).__init__(*args, **kwargs)

	def loadImage(self, file):
		self.img = cv2.imread(file, 0)
		# cv2.imshow("show", self.img)
		# cv2.waitKey(0)
		self.dimensions = self.img.shape
		print("Image Loaded. \nImage Shape: {}".format(self.dimensions))

	def preprocessing(self):
		# STEP 1: convert image to grayscale
		# STEP 2: crop images to appropriate size
		pass

	def cvt2gray(self):
		self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

	def crop(self):
		self.img = self.img[1:100,2:200]

	def generate_spectrograph(self):
		self.flatImage = np.zeros(self.dimensions[1])
		print(self.flatImage.shape)
		for i in range(self.dimensions[1]):
			for j in range(self.dimensions[0]):
				self.flatImage[i] = self.flatImage[i] + self.img[j,i]

		return self.flatImage

	def plot(self):
		plt.plot(self.flatImage)
		plt.show()

def main():

	A = genSpectrogram()
	A.loadImage(r"img.png")
	# A.cvt2gray()
	# F = A.generate_spectrograph()
	# print(F.dtype)
	# np.savetxt("test/chromatograms/FSeries/f3-CBD.csv", F, fmt='%.10f', delimiter=',')
	# cv2.imshow("1", A.img)
	# cv2.waitKey(0)


# def main():
# 	A = genSpectrogram()
# 	A.loadImage("../../../../Downloads/Photos/F1-1.jpg")
# 	A.cvt2gray()
# 	B = A.generate_spectrograph()
# 	np.savetxt("test/chromatograms/FSeries/f1_1.csv", B, fmt='%.10f', delimiter=',')


if __name__ == "__main__":
	main()