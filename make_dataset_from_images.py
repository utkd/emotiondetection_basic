import sys
from os import listdir
from os.path import isfile, join
import numpy as np
import PIL.Image as Image

if len(sys.argv) < 3:
	print "Usage: python <script> <source images folder> <output filename>."
	sys.exit(1)

inputfolderpath = sys.argv[1]

# Get list of all files in the input folder (maybe have a check for image files but eh)
filenames = [f for f in listdir(inputfolderpath) if isfile(join(inputfolderpath, f))]

outfile = open(sys.argv[2],'w')

imgCount = 0

# Edit this line to generate/not generate labels
generateLabels = False

def computeLabel(filename):
	"""
	Depending on the application, generate labels derived from filename
	"""

	#Gender detection
	if filename[1] == 'F':
		return "1"
	else:
		return "0"

	# Emotion Detection
	# if filename[4:6] == "HA":
	# 	return "1"
	# else:
	# 	return "0"

# Go over all the images
for imgfile in filenames:

	if generateLabels:
		classLabel = computeLabel(imgfile)

	# Load the image
	imgPath = join(inputfolderpath, imgfile)
	imgObj = Image.open(imgPath)

	pixelvalues = np.asarray(imgObj.getdata())

	for v in pixelvalues:
		if v > 1000:
			print v, imgfile
			sys.exit(1)
		outfile.write(str(v) + " ")
	if generateLabels:
		outfile.write(classLabel)
	outfile.write("\n")

	imgCount += 1
	if imgCount % 100 == 0:
		print imgCount

outfile.close()