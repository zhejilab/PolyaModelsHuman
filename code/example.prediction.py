#!/usr/bin/env python
# coding: utf-8

#############################
## Imports
#############################

## Import modules for model construction and preparation

from prediction_utilities import input_generator

## Required inputs

import os, tqdm, logging, argparse
import numpy as np
import tensorflow as tf

from keras.models import load_model

logging.basicConfig(level=logging.DEBUG)



#############################
## Helper functions
#############################

def make_predictions(header, lines, outfile, modeltype):

	logging.info(f"Making batch predictions for compiled input data: sequences={len(lines['sequences'])}.")

	Xnew = input_generator.generatedata_sequence(lines['sequences'], 'predict')['predict'][0][0]
	Ypred = model.predict(Xnew, verbose=0)
	logging.info("Predictions complete, recording results.")


	if (modeltype.lower() == "polyaid"):

		classprobs = Ypred[0].flatten().tolist()
		cleavage_vectors = Ypred[1]

		for l,mc,cp,cv in tqdm.tqdm(zip(lines['lines'], lines['max_cleavages'], classprobs, cleavage_vectors), desc = 'Recording predictions'):

			sub_cv = cv - (1.0/np.shape(cv)[0])
			sub_cv[sub_cv <= 0] = 0
			norm_cv = sub_cv / np.sum(sub_cv) if (np.sum(sub_cv) > 0) else np.asarray([0]*np.shape(cv)[0])

			outline = l + f'\t{mc}\t{cp:.06f}\t{str(cv.round(decimals=6).tolist())}\t{str(norm_cv.round(decimals=6).tolist())}\n'
			outfile.write(outline)


	elif ((modeltype.lower() == "polyaid_classification") or (modeltype.lower() == "polyaclassifier")):
		
		classprobs = Ypred.flatten().tolist()

		for l,mc,cp in tqdm.tqdm(zip(lines['lines'], lines['max_cleavages'], classprobs), desc = 'Recording predictions'):
			outline = l + f'\t{cp:.06f}\n'
			outfile.write(outline)


	elif ((modeltype.lower() == "polyaid_cleavage") or (modeltype.lower() == "polyacleavage")):

		cleavage_vectors = Ypred

		for l,mc,cv in tqdm.tqdm(zip(lines['lines'], lines['max_cleavages'], cleavage_vectors), desc = 'Recording predictions'):

			sub_cv = cv - (1.0/np.shape(cv)[0])
			sub_cv[sub_cv <= 0] = 0
			norm_cv = sub_cv / np.sum(sub_cv) if (np.sum(sub_cv) > 0) else np.asarray([0]*np.shape(cv)[0])

			outline = l + f'\t{mc}\t{str(cv.round(decimals=6).tolist())}\t{str(norm_cv.round(decimals=6).tolist())}\n'
			outfile.write(outline)


	elif (modeltype.lower() == "polyastrength"):

		strengths = Ypred.flatten()

		for l,mc,s in tqdm.tqdm(zip(lines['lines'], lines['max_cleavages'], strengths), desc = 'Recording predictions'):

			outline = l + f'\t{s:.06f}\t{(2**s)/(1+(2**s)):.06f}\n'
			outfile.write(outline)

	outfile.flush()

	return



#############################
## Prepare inputs and outputs
#############################

## Define arguments

parser = argparse.ArgumentParser(description = '---')

parser.add_argument('-m', '--model',     type = str, default = 'NA', help = "Path to trained model weights file that will be used to make predictions")
parser.add_argument('-t', '--modeltype', type = str, default = 'NA', help = "Type of model that will be used to make predictions")
parser.add_argument('-d', '--data',      type = str, default = 'NA', help = "Path to dataset we want predictions for")
parser.add_argument('-n', '--dataname',  type = str, default = None, help = "Nickname for dataset")
parser.add_argument('-o', '--outdir',    type = str, default = 'NA', help = "Path to output directory")

args = parser.parse_args()



## Define inputs and outputs

os.makedirs(args.outdir, exist_ok = True)

if (args.dataname is not None):
	outfile_name = os.path.join(args.outdir, f"{args.dataname}.predictions_{args.modeltype}.txt")
else:
	outfile_name = os.path.join(args.outdir, os.path.basename(args.data).replace(".txt", f".predictions_{args.modeltype}.txt"))

logging.info(f"Recording results to: {outfile_name}")



## Prepare model for predictions


from keras.layers import LSTM, Bidirectional

model = load_model(
    args.model,
    custom_objects={"LSTM": LSTM, "Bidirectional": Bidirectional}, 
	compile=False
)

logging.info(f"Model loaded from: {args.model}")



#############################
## Compile sequences from input dataset and make batch-wise predictions
#############################

with open(outfile_name, mode = 'w') as outfile:
	with open(args.data, mode = 'r') as infile:
		header = infile.readline().rstrip("\n").split("\t")

		## Carry over header from input and add output prediction labels

		if (args.modeltype.lower() == "polyaid"):
			outfile.write("\t".join(header) + '\tmax_cleavage\tclassification\tcleavage_vector\tcleavage_norm\n')

		elif ((args.modeltype.lower() == "polyaid_classification") or (args.modeltype.lower() == "polyaclassifier")):
			outfile.write("\t".join(header) + '\tclassification\n')

		elif ((args.modeltype.lower() == "polyaid_cleavage") or (args.modeltype.lower() == "polyacleavage")):
			outfile.write("\t".join(header) + '\tmax_cleavage\tcleavage_vector\tcleavage_norm\n')

		elif (args.modeltype.lower() == "polyastrength"):
			outfile.write("\t".join(header) + '\tpred_logit\tpred_prob\n')

		## Iteratively process input data and compile sequences for batch predictions
		
		batch_size    = 1e4
		batch_counter = 0
		batch_lines   = {'lines':[], 'sequences':[], 'max_cleavages':[]}

		for line in infile:

			## Extract the input sequence and pad with Ns to length according to the trained model parameters.
			ldict = dict(zip(header, line.rstrip("\n").split("\t")))
			lseq  = ldict['sequence'].upper()

			if (len(lseq) != 240):
				raise ValueError(f"Input sequence size incorrect: expected=240, input={len(lseq)}, sequence={lseq}, ldict={ldict}.")

			## If available, extract the input cleavage vector and trim to a standardized window length (default 50 nt).
			## Otherwise, the max cleavage position will be set to a missing value.
			max_cleavage = np.nan

			## Record sequences

			batch_lines['lines'].append(line.rstrip("\n"))
			batch_lines['sequences'].append(lseq)
			batch_lines['max_cleavages'].append(max_cleavage)
			batch_counter += 1

			if (batch_counter == batch_size):
				
				make_predictions(header, batch_lines, outfile, args.modeltype)
				
				batch_lines   = {'lines':[], 'sequences':[], 'max_cleavages':[]}
				batch_counter = 0

		## Make predictions for final batch

		if (batch_counter):
			
			make_predictions(header, batch_lines, outfile, args.modeltype)




