# PolyaModels

This repository contains the necessary scripts to make predictions using **PolyaID** and **PolyaStrength**, convolutional neural network models that predict the classification and cleavage profile surrounding a putative polyA site and its strength, respectively. We developed PolyaID and PolyaStrength to identify putative polyA sites across the genome at nucleotide-resolution and then quantify the cis-regulatory and genomic context factors governing site usage. Because of the large size of the PolyaID and PolyaStrength models, we recommend downloading them as a ZIP file, as they will not be properly downloaded during repo cloning.

Contact *zhe.ji (at) northwestern.edu* with any questions.


## Making New Predictions

### Running PolyaModels requires the following packages be installed

- Python == 3.9.6
- Tensorflow == 2.19.0
- Keras == 3.10.0
- NumPy == 2.0.2
- Pandas == 2.2.3
- pyfaidx == 0.8.1.4
- Isolearn == 0.2.1
- Matplotlib == 3.9.4
- Ipykernel == 6.29.5

Warning: We are still validating that these are the only required packages. 

All versions were optimally selected by pip.

### Usage directions for current pipeline - jupyter notebook
PolyaID and PolyaStrength can be used to make new predictions from new sequences, or a file containing genomic regions of interest. Due to biological relevancy and model interpertability, we do not support analysis of sequences shorter than 60 nucleotides.

#### Important Notice
Due to file sizes, the genome FASTA files could not be added to this repo. If you wish to use these files for predictions please name them as they appear in the jupyter notebook workflow in the cell under the header 'Data'. The missing files are:
'resources/hg38.genome.fa' - genome fasta file  
'resources/hg38.fa.fai' - genome fasta index  
'resources/hg38.chrom.sizes' - chromosome sizes file


**polyA_prediction_pipeline.ipynb**
This file is a jupyter notebook containing cells that allow you to obtain PolyaID and PolyaStrength predictions for genetic sequences and genomic regions of interest. Within the same jupyter notebook, you can visualize these predictions. Here are the steps to run this analysis:
> Important: Assure all paths are correctly set and the jupyter notebook is in the root directory of this repository.  
> Run all helper functions in cells following the header 'Helper Function'.  
> Import data by running cells after header 'Data'.  
> If you have a sequence: begin running analysis at the cell titled 'Input is a sequence'. Change variable named 'seq' under the header 'Initialize your sequence' to your sequence of interest. N.B. ensure you still execute all helper functions at the beginning of the notebook.  
> If you have a .txt file with genomic regions of interest: begin running analysis at cell titled 'Input is txt file'. N.B. All fields in your .txt must be labeled exactly the same as the fields in the provided example file titled 'regions.txt'. In fact, the file must also be named 'regions.txt' to run the pipeline out-of-box. We recommend simply editing the provided .txt file directly.  

**regions.txt**
Text file that works as an input for predicting polya sites along genomic regions of interest given genomic coordinates. An example entry is provided for you. Do not rename the file or rename any columns, the analysis will fail.

**sliding_windows.txt**
File generated if input is a sequence. The sequence is padded and broken up into consecutive windows of 240 nucleotides.


### Usage directions for old Polya Analsyis V1 - Included in Polya_v1 branch

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

**Note:** Predictions can be made from genomic locations but the genome FASTA and chrom.sizes files will need to be provided by the user.

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

We made predictions across hg38 in gene-associated regions and created a compendium of putative polyA sites. These tracks are included when the repository is cloned locally but not when the zipped repository is downloaded. In that case, the tracks will need to be downloaded manually from the GitHub browser using the "Download" button. 

**putative_sites/putative_polya_sites.+.bb**
> Contains the putative polyA sites located on the forward strand in bigBed format. The sites are annotated with the PolyaID classification probability, PolyaStrength score, and gene and feature information.

**putative_sites/putative_polya_sites.-.bb**
> Contains the putative polyA sites located on the reverse strand in bigBed format. The sites are annotated with the PolyaID classification probability, PolyaStrength score, and gene and feature information.


