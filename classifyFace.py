import sys
import numpy as np
import theano
import theano.tensor as T
from finetuneNetwork import StackedDenoisingAutoencoder
import PIL.Image as Image
import cPickle
import lasagne
from os import listdir

if len(sys.argv) < 2:
	print "Usage:", sys.argv[0], "<face image file>"
	sys.exit(1)

#EMOTION_PARAMS_FILE = "finetune.1.4k2k1k250.01.param.happysad"
EMOTION_PARAMS_FILE = "i4.param"
GENDER_PARAMS_FILE = "i4.param.gender"

# Make the network
hidden_layer_sizes = [4000, 2000, 1000, 250]
output_size = 1

input_var = T.dmatrix('inputs')
target_var = T.ivector('targets')

sda = StackedDenoisingAutoencoder(input_var=input_var, hidden_layer_sizes=hidden_layer_sizes, output_size=output_size)
stackedNetwork = sda.getStackedNetwork()
	
f = open(GENDER_PARAMS_FILE,'rb')
all_params = cPickle.load(f)
f.close()
lasagne.layers.set_all_param_values(stackedNetwork, all_params)

#print "\nParams loaded.\n"

imgfilename = sys.argv[1]
imgObj = Image.open(imgfilename)
grayscaleImg = imgObj.convert('L')
resizedImg = grayscaleImg.resize((112, 152))
resizedImg.save("edited.jpg")

pixelvalues = np.asarray(resizedImg.getdata())
pixelvalues = np.reshape(pixelvalues, (1,112*152))
pixelvalues = pixelvalues / np.float32(256.)

prediction = lasagne.layers.get_output(stackedNetwork, deterministic=True)[:,0]

network_prediction = theano.function(inputs=[input_var], outputs=[prediction])
#print pixelvalues.shape
emotionValue =  network_prediction(pixelvalues)[0]

#print "Loading params from", GENDER_PARAMS_FILE

sda2 = StackedDenoisingAutoencoder(input_var=input_var, hidden_layer_sizes=hidden_layer_sizes, output_size=output_size)
stackedNetwork = sda2.getStackedNetwork()

f = open(EMOTION_PARAMS_FILE,'rb')
all_params = cPickle.load(f)
f.close()
lasagne.layers.set_all_param_values(stackedNetwork, all_params)

prediction = lasagne.layers.get_output(stackedNetwork, deterministic=True)[:,0]

network_prediction = theano.function(inputs=[input_var], outputs=[prediction])
genderValue =  network_prediction(pixelvalues)[0]

print genderValue, emotionValue

emotionText = ""
if emotionValue > 0.60:
	emotionText = "HAPPY"
elif emotionValue > 0.40:
	emotionText = "a LITTLE SAD"
else:
	emotionText = "SAD"

genderText = ""
if genderValue > 0.55:
	genderText = "GIRL"
elif genderValue < 0.45:
	genderText = "GUY"
else:
	genderText = "PERSON"

print "The", genderText, "looks", emotionText