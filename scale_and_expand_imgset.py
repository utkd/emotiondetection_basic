import sys
from os import listdir
from os.path import isfile, join
import numpy as np
import PIL.Image as Image

# Arg1 : Folder containing images to scale
# Arg2 : Folder to put scaled images and new generated images

if len(sys.argv) < 3:
	print "Usage: python <script> <source images folder> <output folder>."
	print "Output folder should be created beforehand. Will contain scaled versions of original images and new images"
	sys.exit(1)

# Get input folder path and output folder path
inputfolderpath = sys.argv[1]
outputfolderpath = sys.argv[2]

# Get list of all files in the input folder (maybe have a check for image files but eh)
filenames = [f for f in listdir(inputfolderpath) if isfile(join(inputfolderpath, f))]

# Fo over all the images
for imgfile in filenames:

	# Load the image
	imgPath = join(inputfolderpath, imgfile)
	imgObj = Image.open(imgPath)

	# Resize the image and convert to grayscale
	imgW, imgH = imgObj.size
	resizedImg = imgObj.resize((imgW/5, imgH/5))
	grayscaleImg = resizedImg.convert("L")

	# Save the grayscale image
	grayscaleImg.save(join(outputfolderpath, imgfile))

	# Get pixel values and construct a flipped image
	pixelvalues = list(grayscaleImg.getdata())
	pixelarray = np.array(pixelvalues,dtype='uint8').reshape(grayscaleImg.size[1], grayscaleImg.size[0])
	reconImg = Image.fromarray(np.fliplr(pixelarray)).convert("L")

	# Construct a new filename and save the flipped image
	newfilename = imgfile[:-4]+"FLIP"+imgfile[-4:]
	reconImg.save(join(outputfolderpath, newfilename))	