# tfbs-NN
## Constructing a pipeline for the creation of a TFBS dataset and the training of a NN 

Present progress:

Notebooks

  1. 2bed.ipynb is used for converting tsv and bed files in the correct format for <bedtools intersect> and costructing the true_posititves and true_negatives of the training dataset
  2. experiment_biosample_hist.ipynb is used for plotting a histogram of the number of experiments of every TF in ENCODE

Scripts

  1. conver_fimo.py , script to convert the ouput of fimo in the correct format for <bedtools intersect>
  2. conert_biomart.py, script to convert the mart_exort.txt file derived from Ensmble Regulatory Regions in the correct format for <bedtools intersect>
  3. filter_fimo.py, script that filters fimo results with the Ensemble Regulatory Regions and Chip-seq experiments and creates a dataset of positive and negative sequences for every TF
  4. run_filter_fimo.sh, bash scripts that runs filter_fimo.py for selected list of TFsS
