import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import gzip
import cPickle

class Autoencoder(object):

	def __init__(self, n_in, n_hid, layer_input, W, b):

		l_in = lasagne.layers.InputLayer(shape=(None, n_in), input_var=layer_input)

		l_hid = lasagne.layers.DenseLayer(l_in, n_hid, W=W, b=b, nonlinearity=lasagne.nonlinearities.sigmoid)

		l_out = lasagne.layers.DenseLayer(l_hid, n_in, W=W.get_value().T, nonlinearity=lasagne.nonlinearities.sigmoid)

		self.__autoencoder = l_out

		prediction = lasagne.layers.get_output(self.__autoencoder)
		loss = lasagne.objectives.squared_error(prediction, layer_input)
		self.__loss = loss.mean()

		params = lasagne.layers.get_all_params(self.__autoencoder, trainable=True)
		self.__updates = lasagne.updates.nesterov_momentum(self.__loss, params, learning_rate=0.001, momentum=0.9)

	def getLossUpdates(self):
		return self.__loss, self.__updates

	def getAutoencoder(self):
		return self.__autoencoder


class StackedDenoisingAutoencoder(object):

	def __init__(self, input_var=None, input_layer_size=152*112, hidden_layer_sizes=None, output_size=1):

		self.__autoencoders = []

		l_in = lasagne.layers.InputLayer(shape=(None, input_layer_size), input_var=input_var)
		
		stacked_network = l_in

		for idx in range(len(hidden_layer_sizes)):
			hid_layer = lasagne.layers.DenseLayer(stacked_network, hidden_layer_sizes[idx], nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotUniform())

			prev_layer_output = lasagne.layers.get_output(stacked_network, deterministic=True)
			if idx == 0:
				prev_layer_size = input_layer_size
			else:
				prev_layer_size = hidden_layer_sizes[idx-1]
			stacked_network = hid_layer

		stacked_network = lasagne.layers.DenseLayer(stacked_network, output_size, nonlinearity=lasagne.nonlinearities.sigmoid)
		self.__stackedNetwork = stacked_network

	def getAutoencoders(self):
		return self.__autoencoders

	def getStackedNetwork(self):
		return self.__stackedNetwork

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	"""
	Iterates over training data in mini batches. Taken from lasagne tutorial
	"""
	assert len(inputs) == len(targets)
	
	if shuffle:
		indices = np.arange(len(inputs))
		#indices = np.arange(num_rows)
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs)-batchsize+1, batchsize):
		if shuffle:
			excerpt = indices[start_idx : start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]

def load_data(dataset):
	"""
	Loads the MNIST data
	Slightly modified version of theano tutorial's load data. Does not create shared variables
	"""
	data = np.loadtxt(dataset,dtype=np.float32)
	np.random.shuffle(data)
	features = data[:,:-1]/np.float32(256.)
	labels = data[:,-1].astype('int32')
	num_data = len(data)
	if num_data == 1960:
		num_train = 1568
		return features[:num_train], labels[:num_train], features[num_train:], labels[num_train:]
	elif num_data == 560:
		num_train = 442
		return features[:num_train], labels[:num_train], features[num_train:], labels[num_train:]

def main(in_filename=None, out_filename=None):
	"""
	Function to train each autoencoder and then fine tune the entire network
	"""
	num_epochs=10
	hidden_layer_sizes = [4000, 2000, 1000, 250]
	output_size = 1


	print "Loading data .."
	X_train, y_train, X_val, y_val = load_data('/home/utdesai/projects/deeplearning/emotiondetection/scripts/emotion_train_data')
	print "Done."

	print X_train.shape, y_train.shape, X_val.shape, y_val.shape

	batchsize = len(X_val) / 8

	#input_var = T.tensor4('inputs')
	input_var = T.dmatrix('inputs')
	target_var = T.ivector('targets')

	sda = StackedDenoisingAutoencoder(input_var=input_var, hidden_layer_sizes=hidden_layer_sizes, output_size=output_size)
#	autoencoders = sda.getAutoencoders()
	stackedNetwork = sda.getStackedNetwork()

	print "Loading params from", in_filename

	f = open(in_filename)
	all_params = cPickle.load(f)
	f.close()
	lasagne.layers.set_all_param_values(stackedNetwork, all_params)

	print "\nAutoencoders loaded.\n"

	prediction = lasagne.layers.get_output(stackedNetwork)[:,0]
	networkloss = lasagne.objectives.binary_crossentropy(prediction, target_var)
	networkloss = networkloss.mean()

	networkparams = lasagne.layers.get_all_params(stackedNetwork, trainable=True)
	networkupdates = lasagne.updates.nesterov_momentum(networkloss, networkparams, learning_rate=0.1, momentum=0.9)

	test_prediction = lasagne.layers.get_output(stackedNetwork, deterministic=True)[:,0]
	test_loss = lasagne.objectives.binary_crossentropy(test_prediction, target_var)
	test_loss = test_loss.mean()

	test_acc = T.mean(T.eq(T.round(test_prediction), target_var), dtype=theano.config.floatX)

	#grad_function = theano.function(inputs=[input_var, target_var], outputs=gradients)
	network_prediction = theano.function(inputs=[input_var, target_var], outputs=[prediction, networkloss, target_var])
	network_train_function = theano.function(inputs=[input_var, target_var], outputs=networkloss, updates=networkupdates)
	network_validation_function = theano.function(inputs=[input_var, target_var], outputs=[test_loss, test_acc])

	print "Starting fine tuning .."

	best_acc = 0.6
	best_params = []

	for epoch in range(num_epochs):
		train_err = 0
		train_batches = 0
		start_time = time.time()

		for batch in iterate_minibatches(X_train, y_train, batchsize, shuffle=True):
			inputs, targets = batch

			err = network_train_function(inputs, targets)
			train_err += err
			#print er

			train_batches += 1

		val_err = 0
		val_acc = 0.
		val_batches = 0
		for batch in iterate_minibatches(X_val, y_val, batchsize, shuffle=False):
			inputs, targets = batch
			err, acc = network_validation_function(inputs, targets)
			pred, l, t= network_prediction(inputs, targets)
			val_err += err
			val_acc += acc
			val_batches += 1

		print "Epoch", epoch+1,"of", num_epochs, "took", time.time()-start_time,
		print "training loss", train_err/train_batches,
		print "validation loss", val_err/val_batches,
		print "validation accuracy", val_acc/val_batches

		if val_acc/val_batches > best_acc:
			best_acc = val_acc/val_batches
			best_params = lasagne.layers.get_all_param_values(stackedNetwork)

	print "Fine tuning complete"

	all_params = lasagne.layers.get_all_param_values(stackedNetwork)
	f = open(out_filename+".best",'wb')
	cPickle.dump(best_params, f, protocol=-1)
	f.close()

	f = open(out_filename,'wb')
	cPickle.dump(all_params, f, protocol=-1)
	f.close()
	# test_err = 0
	# test_acc = 0
	# test_batches = 0
	# for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
	# 	inputs, targets = batch
	# 	err, acc = network_validation_function(inputs, targets)
	# 	test_err += err
	# 	test_acc += acc
	# 	test_batches += 1

	# print "\nFinal Result:"
	# print "test loss", test_err/test_batches
	# print "test accuracy", test_acc/test_batches

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "Insufficient Args: python", sys.argv[0], "<input filename> <output filename>"
		print "Recommended filename format: finetune.<iternum>.<hiddenlyr sizes>.<learningrate>.param"
		sys.exit(1)
	main(in_filename=sys.argv[1], out_filename=sys.argv[2])