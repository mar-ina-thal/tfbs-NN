#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os
import argparse

def main(indir, outdir):
    indir = os.path.abspath(indir)  # Convert input directory to absolute path
    outdir = os.path.abspath(outdir)  # Convert output directory to absolute path

    file_path = os.path.join(indir, 'mart_export.txt')
    mart_df = pd.read_csv(file_path, sep='\t')
    mart_df.dropna(inplace=True)

    mapping_table = {
        '1': 'chr1',
        '2': 'chr2',
        '3': 'chr3',
        '4': 'chr4',
        '5': 'chr5',
        '6': 'chr6',
        '7': 'chr7',
        '8': 'chr8',
        '9': 'chr9',
        '10': 'chr10',
        '11': 'chr11',
        '12': 'chr12',
        '13': 'chr13',
        '14': 'chr14',
        '15': 'chr15',
        '16': 'chr16',
        '17': 'chr17',
        '18': 'chr18',
        '19': 'chr19',
        '20': 'chr20',
        '21': 'chr21',
        '22': 'chr22',
        'X': 'chrX',
        'Y': 'chrY',
        'MT': 'chrM'
    }
    mart_df['Chromosome/scaffold name'] = mart_df['Chromosome/scaffold name'].map(mapping_table)

    output_file_path = os.path.join(outdir, 'mart.bed')
    mart_df.to_csv(output_file_path, sep='\t', index=False, header=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Ensmbl data')
    parser.add_argument('--indir', help='Input directory path')
    parser.add_argument('--outdir', help='Output directory path')
    
    args = parser.parse_args()
    main(args.indir, args.outdir)

