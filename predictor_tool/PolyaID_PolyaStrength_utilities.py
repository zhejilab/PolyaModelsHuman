#!/usr/bin/env python
# coding: utf-8


## IMPORT AND SUPPRESS TF WARNINGS

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



## MODEL PARAMETERS

input_shape               = (240,4)

polyaid_bin_len           = 1
polyaid_bin_dropout       = 0.7
polyaid_vec_len           = 50
polyaid_vec_dropout       = 0.1

polyastrength_bin_len     = 1
polyastrength_bin_dropout = 0.1



## HELPER FUNCTIONS

def load_genome(genome_path):
	'''Loads searchable and indexed genome from absolute path for genome FASTA file.
	'''

	import pyfaidx
	return pyfaidx.Fasta(genome_path, sequence_always_upper=True)



def get_chrom_size(chrom_size_path):
	'''Creates dictionary of chromosome sizes in nucleotides.
	'''

	chrom_dict = {}

	with open(chrom_size_path, mode = 'r') as infile:
		for line in infile:
			entries = line.strip("\n").split("\t")
			chrom_dict[entries[0]] = int(entries[1])

	return chrom_dict



def extract_sequence(genome, chrom, start, end, strand):
	'''Use genomic coordinates and indexed genome to extract the genomic sequence for the indicated interval.
	'''

	sequence = genome[chrom][start:end]

	if (strand == "+"):
		return sequence.seq.upper()
	elif (strand == '-'):
		return sequence.reverse.complement.seq.upper()



def get_window(genome, chrom_sizes, chrom, position, strand):
	'''Fetches 240 nt genomic sequence surrounding the indicated position.
	'''

	start = int(position - 120)
	end = int(position + 120)

	if (start <= 0) | (end >= chrom_sizes[chrom]):
		raise ValueError(f'Requested input with interval at ({chrom}:{start}-{end}:{strand}) exceeds the chromosome edges.')

	## if on the reverse strand, shift the coordinates one to put the position of interest at the 120th index in the sequence

	if (strand == '-'):
		start += 1
		end += 1

	return extract_sequence(genome, chrom, start, end, strand).upper()



## CONSTRUCT MODELS AND INPUT DATA

def make_polyaid_model(model_path):
	'''Builds the PolyaID model and loads the trained weights.
	'''

	from contextlib import redirect_stderr

	with redirect_stderr(open(os.devnull, "w")):

		from keras import Input
		from keras.models import Model
		from keras.layers import Dense, Dropout, Flatten, Bidirectional
		from keras.layers import Conv1D, MaxPooling1D, LSTM
		from keras.activations import sigmoid
		from keras.layers.advanced_activations import ReLU	
		from keras import backend as K

	import tensorflow as tf
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	
	model_input = Input(shape = input_shape)
	
	x = ReLU()(Conv1D(512, 8, padding = 'valid', strides = 1, name = 'bin_conv')(model_input))
	x = MaxPooling1D(pool_size = 3, strides = 3, name = 'bin_pool')(x)
	x = Dropout(polyaid_bin_dropout)(x)
	x = Bidirectional(LSTM(units = 128, return_sequences = True, name = 'bin_lstm'))(x)
	x = Dropout(polyaid_bin_dropout)(x)
	x = Flatten(name = 'bin_flatten')(x)
		
	bin_x = ReLU()(Dense(256, kernel_initializer = 'glorot_uniform', name = 'bin_dense1')(x))
	bin_x = Dropout(polyaid_bin_dropout)(bin_x)
	bin_x = ReLU()(Dense(256, kernel_initializer = 'glorot_uniform', name = 'bin_dense2')(bin_x))
	bin_x = Dropout(polyaid_bin_dropout)(bin_x)
	bin_x = ReLU()(Dense(128, kernel_initializer = 'glorot_uniform', name = 'bin_dense3')(bin_x))
	bin_x = Dropout(polyaid_bin_dropout)(bin_x)
	bin_x = ReLU()(Dense(64, name = 'bin_dense4')(bin_x))
	bin_x = Dense(polyaid_bin_len, activation = 'sigmoid', name = 'bin_predictions')(bin_x)
		
	prob_x = ReLU()(Dense(256, kernel_initializer = 'glorot_uniform', name = 'prob_dense1')(x))
	prob_x = Dropout(polyaid_vec_dropout)(prob_x)
	prob_x = ReLU()(Dense(256, kernel_initializer = 'glorot_uniform', name = 'prob_dense2')(prob_x))
	prob_x = Dropout(polyaid_vec_dropout)(prob_x)
	prob_x = ReLU()(Dense(128, kernel_initializer = 'glorot_uniform', name = 'prob_dense3')(prob_x))
	prob_x = Dropout(polyaid_vec_dropout)(prob_x)
	prob_x = ReLU()(Dense(64, name = 'prob_dense4')(prob_x))
	prob_x = Dense(polyaid_vec_len, activation = 'softmax', kernel_initializer = 'zeros', name = 'prob_predictions')(prob_x)
	
	model = Model(inputs = [model_input], outputs = [bin_x, prob_x], name = 'model')
	model.load_weights(model_path)
		
	return model



def make_polyastrength_model(model_path):
	'''Builds the PolyaStrength model and loads the trained weights.
	'''

	from contextlib import redirect_stderr

	with redirect_stderr(open(os.devnull, "w")):

		from keras import Input
		from keras.models import Model
		from keras.layers import Dense, Dropout, Flatten, Bidirectional
		from keras.layers import Conv1D, MaxPooling1D, LSTM
		from keras.layers.advanced_activations import ReLU	
		from keras import backend as K

	import tensorflow as tf
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	model_input = Input(shape = input_shape)

	x = ReLU()(Conv1D(64, 8, padding = 'valid', strides = 1, name = 'seq_conv')(model_input))
	x = MaxPooling1D(pool_size = 3, strides = 3, name = 'bin_pool')(x)
	x = Dropout(polyastrength_bin_dropout)(x)
	x = Bidirectional(LSTM(units = 16, return_sequences = True, name = 'bin_lstm'))(x)
	x = Dropout(polyastrength_bin_dropout)(x)
	x = Flatten(name = 'bin_flatten')(x)
	
	bin_x = ReLU()(Dense(128, kernel_initializer = 'glorot_uniform', name = 'bin_dense1')(x))
	bin_x = Dropout(polyastrength_bin_dropout)(bin_x)
	bin_x = ReLU()(Dense(64, name = 'bin_dense2')(bin_x))
	bin_x = Dense(polyastrength_bin_len, activation = 'linear', name = 'bin_predictions')(bin_x)
	
	model = Model(inputs = model_input, outputs = bin_x, name = 'model')
	model.load_weights(model_path)
	
	return model



def generate_data(sequences):
	'''Prepares data generator to flow data into the models for prediction.
	'''

	import numpy as np
	import pandas as pd
	import isolearn.keras as iso

	if (isinstance(sequences, str)):
		sequences = [sequences]

	df = pd.DataFrame.from_dict({'Sequence' : sequences})
	data_idx = np.arange(len(df), dtype = np.int)
	
	allSequenceGenerator = {
			gen_id : iso.DataGenerator(
				idx,
				{
					'df' : df
				},
				batch_size = len(idx),
				inputs = [
					{
						'id'          : 'seq',
						'source_type' : 'dataframe',
						'source'      : 'df',
						'extractor'   : lambda row,index: row['Sequence'].upper(),
						'encoder'     : iso.OneHotEncoder(seq_length = input_shape[0]),
						'dim'         : input_shape,
						'sparsify'    : False,
					},
				],
				randomizers = [],
				shuffle = False,
				densify_batch_matrices = True,
				move_outputs_to_inputs = False
			)
			for gen_id, idx in [('predict', data_idx)]
		}
	return allSequenceGenerator


