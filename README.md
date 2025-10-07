# PolyaModels

This repository contains the necessary scripts to make predictions using **PolyaID** and **PolyaStrength**, convolutional neural network models that predict the classification and cleavage profile surrounding a putative polyA site and its strength, respectively. We developed PolyaID and PolyaStrength to identify putative polyA sites across the genome at nucleotide-resolution and then quantify the cis-regulatory and genomic context factors governing site usage. Because of the large size of the PolyaID and PolyaStrength models, we recommend downloading them as a ZIP file, as they will not be properly downloaded during repo cloning.

Contact *zhe.ji (at) northwestern.edu* with any questions.


## Making New Predictions

### Running PolyaModels requires specific packages to be installed. The dependencies all exist in the "environment.yml" file. Run this command to create a compatible mamba environment:

```bash
mamba env create -f environment.yml 
```

This will produce a mamba environment named "polya_github" that can run the human_pipeline example shown in the directly subsequent steps of this ReadMe. All package versions were optimally selected by mamba and pip.

**`human_pipeline.py`** runs PolyaID & PolyaStrength on sliding windows across a **user-provided sequence** and saves results. It generates per-nucleotide PolyaID predictions, cleavage vectors, and PolyaStrength predictions. Additionally, it procudes a full-window summary plot of all predictions.

## Usage
The example command line below can also be found at results/human_example.command_line.txt

```bash
python human_pipeline.py -s AGAGCCGTGAAGGCCCAGGGGACCTGCGTGTCTTGGCTCCACGCCAGATGTGTTATTATTTATGTCTCTGAGAATGTCTGGATCTCAGAGCCGAATTACAATAAAAACATCTTTAAACTTATTTCTACCTCATTTTGGGGTTGCCAGCTCACCTGATCATTTTTATGAACTGTCATGAACACTGATGACATTTTATGAGCCTTTTACATGGGACACTACAGAATACATTTGTCAGCGAGG
```
**Note: Put in the the sequence directly** (no quotes) and press Enter.       
**Example sequence**: 
```
AGAGCCGTGAAGGCCCAGGGGACCTGCGTGTCTTGGCTCCACGCCAGATGTGTTATTATTTATGTCTCTGAGAATGTCTGGATCTCAGAGCCGAATTACAATAAAAACATCTTTAAACTTATTTCTACCTCATTTTGGGGTTGCCAGCTCACCTGATCATTTTTATGAACTGTCATGAACACTGATGACATTTTATGAGCCTTTTACATGGGACACTACAGAATACATTTGTCAGCGAGG
```
**Allowed characters:** `A C G T` only. Any other character will raise an error.

**Note:** The first 30 nucleotides and last 30 nucleotides of your sequence can not be predicted as representative polya sites by this pipeline. Due to padding sequences with 'N' nucleotides to generate scanning windows, boundary nucleotides have predictions that can be unduly influenced by 'N' nucleotides.

## Outputs

- **`results/cleavage_profile_explanation.human_example.svg`** - Comprehensive predictions plot to identify individual Polya sites

[![polya_cleavage_profiles](/results/cleavage_profile_explanation.human_example.svg)]

``` 
Axis1: PolyaID classification – Predicted polyadenylation probability per position (cutoff = 0.75)
Axis2: Positive cleavage vectors – Distribution of cleavage site predictions.
Axis3: Normalized cleavage profile – Cleavage probability across all scanning model predictions.
Axis4: Representative cleavage site – Most likely polyadenylation sites.
```  
- **`results/polya_sites.human_example.txt`** – Each identified representative polya site:  
```
Position: 121
PolyaID: 0.99
PolyaStrength: -2.03
sequence: AGAGCCGTGAAGGCCCAGGGGACCTGCGTGTCTTGGCTCCACGCCAGATGTGTTATTATTTATGTCTCTGAGAATGTCTGGATCTCAGAGCCGAATTACAATAAAAACATCTTTAAACTTATTTCTACCTCATTTTGGGGTTGCCAGCTCACCTGATCATTTTTATGAACTGTCATGAACACTGATGACATTTTATGAGCCTTTTACATGGGACACTACAGAATACATTTGTCAGCGAGG
cleavage_vector: [0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.015632,0.069378,0.390414,0.362004,0.132278,0.029009,0.001284,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000]
```

## Jupyter Notebook Polya Prediction Workspace  

### Usage directions for current pipeline - jupyter notebook
PolyaID and PolyaStrength can be used to make new predictions from new sequences, or a file containing genomic regions of interest. Due to biological relevancy and model interpertability, we do not support analysis of sequences shorter than 60 nucleotides.

```
If you want to customize output files or output plots, the functions used in pipeline.py are all laid out in an easy to interpret fashion across different jupyter notebook cells.
```

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
