# PolyaModels

This repository contains the necessary scripts to make predictions using **PolyaID** and **PolyaStrength**, convolutional neural network models that predict the classification and cleavage profile surrounding a putative polyA site and its strength, respectively. We developed PolyaID and PolyaStrength to identify putative polyA sites across the genome at nucleotide-resolution and then quantify the cis-regulatory and genomic context factors governing site usage.

Contact *zhe.ji (at) northwestern.edu* with any questions.


## Making New Predictions

### Running PolyaModels requires the following packages be installed

- Python == 3.6
- Tensorflow == 2.1.0
- Keras == 2.3.1
- NumPy == 1.19.1
- Pandas == 1.1.5
- pyfaidx == 0.5.9
- Isolearn

### Usage

PolyaID and PolyaStrength can be used to make new predictions from a genomic location, new sequences, or a file containing either of these inputs. When making individual predictions, genomic locations must be given as a string with the format "chrom:position:strand" and requires that reference genome FASTA and chromosome sizes files be provided. In file format, the genomic locations should be provided as BED6 intervals.

**predictor_tool/PolyaID_PolyaStrength_prediction.py**
> This file contains the predictor tool to make new predictions using PolyaID and PolyaStrength. It is designed to be used as a command-line tool, which users can invoke as shown in the examples below.

**predictor_tool/PolyaID_PolyaStrength_utilities.py**
> This file contains the utility and helper functions used by the predictor tool. For example, functions to extract the genomic sequence if needed, build and reload the PolyaID and PolyaStrength models, and flow batches of data into the models for predictions are here.

**predictor_tool/PolyaID_model.h5**
> The trained model weights for PolyaID.

**predictor_tool/PolyaStrength_model.h5**
> The trained model weights for PolyaStrength.

### Example prediction from sequence

```sh
python PolyaID_PolyaStrength_prediction.py from_sequence -s 'AGAGCCGTGAAGGCCCAGGGGACCTGCGTGTCTTGGCTCCACGCCAGATGTGTTATTATTTATGTCTCTGAGAATGTCTGGATCTCAGAGCCGAATTACAATAAAAACATCTTTAAACTTATTTCTACCTCATTTTGGGGTTGCCAGCTCACCTGATCATTTTTATGAACTGTCATGAACACTGATGACATTTTATGAGCCTTTTACATGGGACACTACAGAATACATTTGTCAGCGAGG'
```

This will give the following output: 

```
Sequence: AGAGCCGTGAAGGCCCAGGGGACCTGCGTGTCTTGGCTCCACGCCAGATGTGTTATTATTTATGTCTCTGAGAATGTCTGGATCTCAGAGCCGAATTACAATAAAAACATCTTTAAACTTATTTCTACCTCATTTTGGGGTTGCCAGCTCACCTGATCATTTTTATGAACTGTCATGAACACTGATGACATTTTATGAGCCTTTTACATGGGACACTACAGAATACATTTGTCAGCGAGG
PolyaID: 0.9993073 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.015632376074790955, 0.06937781721353531, 0.3904140591621399, 0.3620043098926544, 0.132278174161911, 0.029009360820055008, 0.0012839401606470346, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PolyaStrength: -2.0295653
```

A putative polyA site with this sequence is expected to occur with a classification probability >0.999 from PolyaID. The cleavage probability +/- 25 nt surrounding this site is then predicted. This site has a PolyaStrength score of -2.03, which on a probability scale is equivalent to an expected usage of ~20%.

### Example prediction from genomic location

We can make a prediction 

```sh
python PolyaID_PolyaStrength_prediction.py from_position -p 'chr1:932116:+'  -g ./genome.fa -c ./chrom.sizes
```

This will give the following output:

```
Sequence: AGAGCCGTGAAGGCCCAGGGGACCTGCGTGTCTTGGCTCCACGCCAGATGTGTTATTATTTATGTCTCTGAGAATGTCTGGATCTCAGAGCCGAATTACAATAAAAACATCTTTAAACTTATTTCTACCTCATTTTGGGGTTGCCAGCTCACCTGATCATTTTTATGAACTGTCATGAACACTGATGACATTTTATGAGCCTTTTACATGGGACACTACAGAATACATTTGTCAGCGAGG
PolyaID: 0.9993073 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.015632376074790955, 0.06937781721353531, 0.3904140591621399, 0.3620043098926544, 0.132278174161911, 0.029009360820055008, 0.0012839401606470346, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PolyaStrength: -2.0295653
```

At chromosome 1, position 932,116 on the forward strand, a putative polyA site at this location is expected with a classification probability >0.999 from PolyaID. The cleavage probability +/- 25 nt surrounding this site is then predicted. This site has a PolyaStrength score of -2.03, which on a probability scale is equivalent to an expected usage of ~20%.


## Putative PolyA Sites in hg38

We made predictions across hg38 in gene-associated regions and created a compendium of putative polyA sites. 

**putative_sites/putative_polya_sites.+.bb**
> Contains the putative polyA sites located on the forward strand in bigBed format. The sites are annotated with the PolyaID classification probability, PolyaStrength score, and gene and feature information.

**putative_sites/putative_polya_sites.-.bb**
> Contains the putative polyA sites located on the reverse strand in bigBed format. The sites are annotated with the PolyaID classification probability, PolyaStrength score, and gene and feature information.


