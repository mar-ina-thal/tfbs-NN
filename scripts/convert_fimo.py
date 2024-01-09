#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import argparse
import os

def main(indir, outdir):
    indir = os.path.abspath(indir)  # Convert input directory to absolute path
    outdir = os.path.abspath(outdir)  # Convert output directory to absolute path

    file_path = os.path.join(indir, 'fimo.tsv')

    df = pd.read_csv(file_path, sep='\t')
    df.dropna(inplace=True)
    
    #Rename columns
    df.rename(columns={"sequence_name": "Chromosome", "start": "Start", "stop": "End"}, inplace=True)
    df["Start"]=df["Start"].astype('int')
    df["End"]=df["End"].astype('int')
    
    
    # Extracting required information using regular expressions
    pattern = r'(PEAK_\d+)::(chr[^:]+):(\d+)-(\d+)'
    extracted = df['Chromosome'].str.extract(pattern)

    extracted.columns = ['Peak', 'Chromosome', 'Peak Start', 'Peak End']


    extracted['Peak Start'] = pd.to_numeric(extracted['Peak Start'], errors='coerce')
    extracted['Peak End'] = pd.to_numeric(extracted['Peak End'], errors='coerce')
    df['Start'] = df['Start'] + extracted['Peak Start']
    df['End'] = df['End'] + extracted['Peak Start']
    
    # Filter rows that start with 'PEAK'
    df = df[df['Chromosome'].str.startswith('PEAK')]

    # Extract the 'chr' part from the 'Chromosome' column using .loc
    df['Chromosome'] = df['Chromosome'].str.extract(r'(chr[^:]+)')
    
    # Rearange columns
    df['name'] = df['motif_alt_id'].astype(str) + '_' + df['p-value'].astype(str) + '_' + df['q-value'].astype(str) + '_' + df['matched_sequence']
    df.drop(['motif_id', 'p-value', 'q-value', 'matched_sequence', 'motif_alt_id' ], axis=1, inplace=True)
    
    # Keep only known chromosomes
    expected_list = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
                     'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY', 'chrM']
    
    df = df[df['Chromosome'].isin(expected_list)]

    output_file_path = os.path.join(outdir, 'fimo_full.bed')
    df.to_csv(output_file_path, sep='\t', index=False, header=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process FIMO data')
    parser.add_argument('--indir', help='Input directory path')
    parser.add_argument('--outdir', help='Output directory path')
    
    args = parser.parse_args()
    main(args.indir, args.outdir)