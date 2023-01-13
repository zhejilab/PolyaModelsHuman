#!/usr/bin/env python
# coding: utf-8


## IMPORTS

import PolyaID_PolyaStrength_utilities as utils

import os, argparse
import numpy as np



## DEFINE INPUTS

parser     = argparse.ArgumentParser(description = 'Make new predictions using PolyaID and PolyaStrength')
subparsers = parser.add_subparsers(help = 'sub-command help', dest = 'input_type')

parser_file = subparsers.add_parser('from_file', help = 'from file help')
parser_file.add_argument('-i', '--input',          type = str, default = None, help = 'Absolute path to a file containing sites of interest for prediction. Sites should be formatted so that the start and end position are 1 nt apart and mark the center point of interest.')
parser_file.add_argument('-f', '--format',         type = str, choices = ['BED6','sequences'], required = True, help = 'Format of input file. If BED6, sites should be formatted so that the start and end position are 1 nt apart and mark the center point of interest. If providing file of sequences, the sequences should be in a single column.')
parser_file.add_argument('-g', '--genome',         type = str, default = None, help = 'Absolute path to a genome FASTA file containing the chromosome sequences.')
parser_file.add_argument('-c', '--chromsizes',     type = str, default = None, help = 'Absolute path to a file containing the chromosome names and sizes in nucleotides. The file should be two columns.')

parser_position = subparsers.add_parser('from_position', help = 'from position help')
parser_position.add_argument('-p', '--position',   type = str, required = True, help = 'String indicating a single genomic position for prediction. Sites must be formatted as chromosome:position:strand, where the position is the same as the start coordinate in a BED file.')
parser_position.add_argument('-g', '--genome',     type = str, required = True, help = 'Absolute path to a genome FASTA file containing the chromosome sequences.')
parser_position.add_argument('-c', '--chromsizes', type = str, required = True, help = 'Absolute path to a file containing the chromosome names and sizes in nucleotides. The file should be two columns.')

parser_sequence = subparsers.add_parser('from_sequence', help = 'from sequence help')
parser_sequence.add_argument('-s', '--sequence',   type = str, default = None, help = 'String containing 240 nt sequence to be directly used for prediction. This sequence should not contain any N nucleotides.')

args = parser.parse_args()



## PREPARE MODELS

polyaID       = utils.make_polyaid_model("PolyaID_model.h5")
polyaStrength = utils.make_polyastrength_model("PolyaStrength_model.h5")



## MAKE PREDICTIONS FOR INPUT SEQUENCE

if (args.input_type == 'from_sequence'):

	sequence = args.sequence.upper()

	if ('N' in sequence):
		raise ValueError("Input sequence contains missing nucleotides.")

	if (len(sequence) != 240):
		raise ValueError("Input sequence is not 240 nt.")

	encoding = utils.generate_data(sequence)['predict'][0]

	polyaID_prediction     = polyaID.predict(encoding)
	polyaID_classification = polyaID_prediction[0][0][0]
	polyaID_rawcleavage    = polyaID_prediction[1].flatten()

	polyaID_subtracted = polyaID_rawcleavage - 0.02
	polyaID_subtracted[polyaID_subtracted <= 0] = 0
	polyaID_normcleavage = polyaID_subtracted / np.sum(polyaID_subtracted) if (np.sum(polyaID_subtracted) > 0) else np.asarray([0]*50)

	polyaStrength_score = polyaStrength.predict(encoding)[0][0]

	print("Sequence:", sequence)
	print("PolyaID:", polyaID_classification, polyaID_normcleavage.tolist())
	print("PolyaStrength:", polyaStrength_score)



## MAKE PREDICTIONS FOR INPUT POSITION

elif (args.input_type == 'from_position'):

	genome = utils.load_genome(args.genome)
	chrom_sizes = utils.get_chrom_size(args.chromsizes)

	chrom, position, strand = [int(x) if (i == 1) else str(x) for i,x in enumerate(args.position.split(":"))]
	sequence = utils.get_window(genome, chrom_sizes, chrom, position, strand)

	if ('N' in sequence):
		raise ValueError("Input sequence contains missing nucleotides.")

	if (len(sequence) != 240):
		raise ValueError("Input sequence is not 240 nt.")

	encoding = utils.generate_data(sequence)['predict'][0]

	polyaID_prediction     = polyaID.predict(encoding)
	polyaID_classification = polyaID_prediction[0][0][0]
	polyaID_rawcleavage    = polyaID_prediction[1].flatten()

	polyaID_subtracted = polyaID_rawcleavage - 0.02
	polyaID_subtracted[polyaID_subtracted <= 0] = 0
	polyaID_normcleavage = polyaID_subtracted / np.sum(polyaID_subtracted) if (np.sum(polyaID_subtracted) > 0) else np.asarray([0]*50)

	polyaStrength_score = polyaStrength.predict(encoding)[0][0]

	print("Sequence:", sequence)
	print("PolyaID:", polyaID_classification, polyaID_normcleavage.tolist())
	print("PolyaStrength:", polyaStrength_score)



## MAKE PREDICTIONS FOR INPUT FILE

elif (args.input_type == 'from_file'):

	input_root, input_extension = os.path.splitext(args.input)
	output = input_root + ".with_predictions" + input_extension

	print("Input file:", args.input)
	print("Output file:", output)

	genome = utils.load_genome(args.genome)
	chrom_sizes = utils.get_chrom_size(args.chromsizes)

	# prepare input sequences

	results = {}

	if (args.format == 'BED6'):

		with open(args.input, mode = 'r') as handle:
			for line in handle:

				chrom, start, end, label, score, strand = [int(x) if (i in [1,2]) else float(x) if (i == 4) else str(x) for i,x in enumerate(line.strip().split("\t"))]

				if (end != (start + 1)):
					raise ValueError("Input BED6 file is incorrectly formatted. Intervals must be 1 nt.")

				sequence = utils.get_window(genome, chrom_sizes, chrom, start, strand)
				results[sequence] = line.strip()

	elif (args.format == 'sequences'):

		with open(args.input, mode = 'r') as handle:
			for line in handle:

				sequence = line.strip().upper()
				results[sequence] = line.strip()

	# make predictions

	sequences = list(results.keys())
	encodings = utils.generate_data(sequences)['predict'][0]

	polyaID_predictions     = polyaID.predict(encodings)
	polyaID_classifications = polyaID_predictions[0].flatten()
	polyaID_rawcleavages    = polyaID_predictions[1]

	polyaID_subtracted = polyaID_rawcleavages - 0.02
	polyaID_subtracted[polyaID_subtracted <= 0] = 0
	polyaID_normcleavages = polyaID_subtracted / polyaID_subtracted.sum(axis = 1).reshape(-1,1)

	polyaStrength_scores = polyaStrength.predict(encodings).flatten()
	
	# record results

	with open(output, mode = 'w') as handle:
		for (input_sequence, input_line), polyaID_classification, polyaID_rawcleavage, polyaID_normcleavage, polyaStrength_score in zip(results.items(), polyaID_classifications, polyaID_rawcleavages, polyaID_normcleavages, polyaStrength_scores):
			handle.write(f'{input_line}\t{input_sequence.upper()}\t{polyaID_classification:.06f}\t{str(polyaID_rawcleavage.tolist())}\t{str(polyaID_normcleavage.tolist())}\t{polyaStrength_score:.06f}\n')



