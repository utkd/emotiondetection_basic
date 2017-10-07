import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import gzip
import cPickle

import PIL.Image as Image
from utils import tile_raster_images

class Autoencoder(object):

	def __init__(self, n_in, n_hid, layer_input, W, b):

		l_in = lasagne.layers.InputLayer(shape=(None, n_in), input_var=layer_input)
		l_hid = lasagne.layers.DenseLayer(l_in, n_hid, W=W, b=b, nonlinearity=lasagne.nonlinearities.sigmoid)

		l_out = lasagne.layers.DenseLayer(l_hid, n_in, W=W.T, nonlinearity=lasagne.nonlinearities.sigmoid)

		self.__autoencoder = l_out
		self.__W = W

		prediction = lasagne.layers.get_output(self.__autoencoder)
		loss = lasagne.objectives.squared_error(prediction, layer_input)
		self.__loss = loss.mean()

		params = lasagne.layers.get_all_params(self.__autoencoder, trainable=True)
		self.__updates = lasagne.updates.nesterov_momentum(self.__loss, params, learning_rate=0.01, momentum=0.9)

	def getLossUpdates(self):
		return self.__loss, self.__updates

	def getAutoencoder(self):
		return self.__autoencoder

	def getWeights(self):
		return self.__W

class StackedDenoisingAutoencoder(object):

	def __init__(self, input_var=None, input_layer_size=112*152, output_size=1, hidden_layer_sizes=None):

		self.__autoencoders = []

		l_in = lasagne.layers.InputLayer(shape=(None, input_layer_size), input_var=input_var)
		stacked_network = l_in

		for idx in range(len(hidden_layer_sizes)):
			hid_layer = lasagne.layers.DenseLayer(stacked_network, hidden_layer_sizes[idx], nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

			prev_layer_output = lasagne.layers.get_output(stacked_network, deterministic=True)
			if idx == 0:
				prev_layer_size = input_layer_size
			else:
				prev_layer_size = hidden_layer_sizes[idx-1]
			autoenc = Autoencoder(prev_layer_size, hidden_layer_sizes[idx], prev_layer_output, hid_layer.W, hid_layer.b)
			self.__autoencoders.append(autoenc)

			stacked_network = hid_layer

		stacked_network = lasagne.layers.DenseLayer(stacked_network, output_size, nonlinearity=lasagne.nonlinearities.sigmoid)
		self.__stackedNetwork = stacked_network

	def getAutoencoders(self):
		return self.__autoencoders

	def getStackedNetwork(self):
		return self.__stackedNetwork

def iterate_minibatches(inputs, batchsize, shuffle=False):
	"""
	Iterates over training data in mini batches. Taken from lasagne tutorial
	"""
	
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs)-batchsize+1, batchsize):
		if shuffle:
			excerpt = indices[start_idx : start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt]

def load_data(dataset):
	"""
	Loads the MNIST data
	Slightly modified version of theano tutorial's load data. Does not create shared variables
	"""
	data = np.loadtxt(dataset,dtype=np.float32)
	return data/np.float32(256.)

def main(in_filename=None, out_filename=None):
	"""
	Function to train each autoencoder and then fine tune the entire network
	"""
	num_epochs=25
	batchsize=49
	#hidden_layer_sizes = [6000, 3000, 1000, 300]
	hidden_layer_sizes = [4000, 2000, 1000, 250]
	output_size = 1

	print "Loading data .."
	X_train = load_data('/home/utdesai/projects/deeplearning/emotiondetection/scripts/all.unlabeled.dataset')

	print "Done."

	input_var = T.dmatrix('inputs')

	sda = StackedDenoisingAutoencoder(input_var=input_var, hidden_layer_sizes=hidden_layer_sizes, output_size=1)
	autoencoders = sda.getAutoencoders()
	stackedNetwork = sda.getStackedNetwork()

	if in_filename is not None:
		print "Loading params from", in_filename, ".."
		f = open(in_filename)
		all_params = cPickle.load(f)
		lasagne.layers.set_all_param_values(stackedNetwork, all_params)
		f.close()
		print "Done"

	print "Beginning training .."

	# For each autoencoder, train it for X epochs
	for idx in range(len(autoencoders)):

		print "\nTraining Autoencoder", idx, ".."

		autoencObj = autoencoders[idx]
		autoEncoderNetwork = autoencObj.getAutoencoder()
		loss, updates = autoencObj.getLossUpdates()

		train_function = theano.function(inputs=[input_var], outputs=loss, updates=updates)

		for epoch in range(num_epochs):
			train_err = 0
			train_batches = 0
			start_time = time.time()

			for batch in iterate_minibatches(X_train, batchsize, shuffle=True):
				inputs = batch
				train_err += train_function(inputs)
				train_batches += 1

			print "Epoch", epoch+1, "completed in", (time.time()-start_time) ,"secs. with training loss:", train_err/train_batches

			if idx == 0 and (epoch+1)==num_epochs:
				print "Generating Image for epoch", epoch+1, "..",
				image = Image.fromarray(
				tile_raster_images(X=autoencObj.getWeights().get_value(borrow=True).T, img_shape=(152, 112), tile_shape=(10, 10),tile_spacing=(1, 1)))
				image.save('autoencoder'+str(epoch+1)+'.png')
				print "Done"

	print "\nAutoencoders training complete.\n"

	# Save params
	if out_filename is not None:
		all_params = lasagne.layers.get_all_param_values(stackedNetwork)
		f = open(out_filename,'wb')
		cPickle.dump(all_params, f)
		f.close()
	
if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "Insufficient Args: python", sys.argv[0], "<output filename> <input filename>(optional)"
		print "Recommended filename format: autoenc.<iternum>.<hiddenlyr sizes>.<learningrate>.param"
		sys.exit(1)
	if len(sys.argv) == 2:
		main(in_filename=None, out_filename=sys.argv[1])
	else:
		main(in_filename=sys.argv[2], out_filename=sys.argv[1])