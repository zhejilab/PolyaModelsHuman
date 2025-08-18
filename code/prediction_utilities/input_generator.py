#!/usr/bin/env python

import numpy as np
import pandas as pd
import isolearn.keras as iso


def generatedata_sequence(sequences, data_partition, verbose = False):

	if (isinstance(sequences, str)):
		sequences = [sequences]

	df = pd.DataFrame.from_dict({'sequence' : sequences})
	
	seq_gen = {
		gen_id : iso.DataGenerator(
			idx,
			{
				'df' : df
			},
			batch_size = len(idx),
			inputs = [
				{
					'id' : 'seq',
					'source_type' : 'dataframe',
					'source' : 'df',
					'extractor' : lambda row,index: row['sequence'].upper(),
					'encoder' : iso.OneHotEncoder(seq_length = 240),
					'dim' : (240,4),
					'sparsify' : False,
				},
			],
			outputs = [],
			randomizers = [],
			shuffle = False,
			densify_batch_matrices = True,
			move_outputs_to_inputs = False
		)
		for gen_id, idx in [(data_partition, df.index.to_numpy(dtype = int))]
	}

	if (verbose):
		print("Data sets have been generated, encoded, and formatted.")

	return seq_gen


