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

**`pipeline.py`** runs PolyaID & PolyaStrength on sliding windows around a **user-provided sequence** and saves results. It generates per-window PolyaID, cleavage vectors, PolyaStrength and a summary plot.

## Usage

```bash
python pipeline.py
```

When prompted:

```
Enter your sequence:
```

**Put in the the sequence directly** (no quotes) and press Enter.       
**Example sequence**: 
```
AGAGCCGTGAAGGCCCAGGGGACCTGCGTGTCTTGGCTCCACGCCAGATGTGTTATTATTTATGTCTCTGAGAATGTCTGGATCTCAGAGCCGAATTACAATAAAAACATCTTTAAACTTATTTCTACCTCATTTTGGGGTTGCCAGCTCACCTGATCATTTTTATGAACTGTCATGAACACTGATGACATTTTATGAGCCTTTTACATGGGACACTACAGAATACATTTGTCAGCGAGG
```
**Allowed characters:** `A C G T` only. Any other character will raise an error.

## Outputs

- **`results/cleavage_profile_explanation.pdf`**
``` 
Comprehensive predictions plot to identify individual Polya sites

Axis1: PolyaID classification – Shows predicted polyadenylation probability per position (red = likely site, blue = unlikely).
Axis2: Positive cleavage vectors – Distribution of cleavage site predictions across the sequence.
Axis3: Normalized cleavage profile – Cleavage probability at each nucleotide across all scanning model predictions.
Axis4: Representative cleavage site – Marks the most likely positions where cleavage actually occurs.
```
- **`sliding_windows.txt`** – 240-nt windows generated from your input (with padding by the script).  
- **`sliding_windows_with_scores_and_index.txt`** – Each window plus:  
    - `PolyaID`
    - `cleavage_vector`
    - `PolyaStrength`
    - `Index`       
- Text file entry for first nucleotide in example sequence:      
```
sequence:NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGAGCCGTGAAGGCCCAGGGGACCTGCGTGTCTTGGCTCCACGCCAGATGTGTTATTATTTATGTCTCTGAGAATGTCTGGATCTCAGAGCCGAATTACAATAAAAACATCTTTAAACTT       
PolyaID: 0.5684121251106262     
cleavage_vector: 0.008032,0.022785,0.000000,0.000000,0.039049,0.029630,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.037699,0.000000,0.009000,0.011723,0.000000,0.041911,0.058239,0.025139,0.018369,0.002342,0.025287,0.016424,0.000503,0.003943,0.031650,0.000000,0.011006,0.000000,0.008044,0.000000,0.106411,0.087432,0.081241,0.110331,0.090367,0.029143,0.025577,0.068724      
PolyaStrength: -7.379417419433594       
Index: 1.0      
```
- Text file entry for central nucleotide in example sequence:        
```
sequence:AGAGCCGTGAAGGCCCAGGGGACCTGCGTGTCTTGGCTCCACGCCAGATGTGTTATTATTTATGTCTCTGAGAATGTCTGGATCTCAGAGCCGAATTACAATAAAAACATCTTTAAACTTATTTCTACCTCATTTTGGGGTTGCCAGCTCACCTGATCATTTTTATGAACTGTCATGAACACTGATGACATTTTATGAGCCTTTTACATGGGACACTACAGAATACATTTGTCAGCGAGG      
PolyaID: 0.9993072152137756     
cleavage_vector: 0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.015632,0.069378,0.390414,0.362004,0.132278,0.029009,0.001284,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000      
PolyaStrength: -2.02956485748291        
Index: 121.0   
```     

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
