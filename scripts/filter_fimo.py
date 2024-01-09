#!/usr/bin/env python
# coding: utf-8


import convert_fimo
import convert_biomart
import os
import pandas as pd
from pyfaidx import Fasta
import argparse


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

# list of tfs to be used
tfs=['CTCF', 'JUN', 'NR3C1', 'TEAD4','ZNF274',
     'USF2', 'TCF12', 'SRF', 'NRF1', 'MAZ',
     'HES2', 'FOS', 'CEBPA', 'FOXA1', 'USF1',
     'RELA', 'GABPA', 'ELF1','EGR1', 'ATF3',
     'ZBTB33','SP1', 'CREB1', 'YY1', 'JUND',
     'FOSL2', 'MAX', 'MYC', 'CEBPB', 'REST']


# Run this command in order modify the mart_export.txt file downloaded from ensembl



def main(biomart_indir, tf):   

    print(f"{tf}")
    
    # Define the necessary directories
    home_dir = os.path.expanduser("~")
    
    chipseq_indir = f"/mnt/raid1/thalassini/home/Desktop/ENCODE_CHIP_new/DATA/{tf}/{tf}.peaks"
    chipseq_outdir = f"/mnt/raid1/thalassini/home/filtered_fimo/{tf}/chipseq_filtered/"    
    fimo_indir = f"/mnt/raid1/thalassini/home/Desktop/ENCODE_CHIP_new/10_RUN/{tf}/run_1"
    fimo_outdir = f"filtered_fimo/{tf}"    
    ensemble_outdir = f"/mnt/raid1/thalassini/home/filtered_fimo/{tf}/ensemble_filterted"
    
    # Create the necessary directories
    os.makedirs(ensemble_outdir, exist_ok=True)
    os.makedirs(chipseq_outdir, exist_ok=True)
    os.makedirs(fimo_outdir, exist_ok=True)
    
    # Converting the fimo.tsv
    print("Converting fimo...")
    convert_fimo.main(fimo_indir,fimo_outdir)
    print("Fimo converted")
    
    # Running bedtools intersect with ensemble Regulatory regions
    print("Running bedtools intersect with ensembl...")
    intersect1 = f"bedtools intersect -a {fimo_outdir}/fimo_full.bed -b {biomart_indir}/mart.bed > {ensemble_outdir}/ensemlb_filtered_fimo.bed"
    os.system(intersect1)

    # Set new permissions (read, write, and execute for all)
    new_permissions = 0o777
    os.chmod(f"{ensemble_outdir}/ensemlb_filtered_fimo.bed" , new_permissions)
    print("Ensembl filtered created")
    
    # Running bedtools intersect with chipseq
    print("Running bedtools intersect with chipseq...")
    intersect2= f"bedtools intersect -a {ensemble_outdir}/ensemlb_filtered_fimo.bed -b {chipseq_indir} > {chipseq_outdir}/chipseq_filtered.bed"
    os.system(intersect2)
    print("Chipseq filtered created")
    
    
    ## Create dataframes
    
    # Create fimo df
    df = pd.read_csv(f"{fimo_outdir}/fimo_full.bed", sep='\t')
    columns=['Chromosome', 'Start', 'End', 'strand', 'score', 'name']
    df.columns=columns
    
    
    # Create ensembl filtered df
    filtered_fimo_path = f"{ensemble_outdir}/ensemlb_filtered_fimo.bed"
    filtered_df = pd.read_csv(filtered_fimo_path, sep='\t')
    filtered_df.dropna(inplace=True)
    filtered_df.columns=columns
    
    
    # Create chipseq filterd df
    chipseq_filtered_fimo_path = f"{chipseq_outdir}/chipseq_filtered.bed"
    chipseq_filtered_df = pd.read_csv(chipseq_filtered_fimo_path, sep='\t')
    chipseq_filtered_df.dropna(inplace=True)
    chipseq_filtered_df.columns=columns
    

    
    # Create negative df
    
    # We create the negative df by excluding from fimo tfs the rows that exist in filtered_df
    
    # Merge the DataFrames based on specific columns and keep rows that do not exist in df_filtered
    merged = df.merge(filtered_df, on=['Chromosome', 'Start', 'End', 'strand', 'score', 'name'], how='outer', indicator=True)
    negative_df = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)
    
    # Calculating reduction percentages
    chipseq_reduction = 100 - len(chipseq_filtered_df) * 100 / len(df)
    filtered_reduction = 100 - len(filtered_df) * 100 / len(df)



    # Calculating statistics of the dataset. Creating the stats.txt
    content = f'Statistics for {tf} dataset\n' 
    content += f" The chipseq filtered data is reduced by {chipseq_reduction:.2f}% from initial fimo results\n"
    content += f" The promoter/enhancer filtered data is reduced by {filtered_reduction:.2f}% from initial fimo results\n"
    
    stats_dir = os.path.join(home_dir, f"filtered_fimo/{tf}/stats")
    os.makedirs(stats_dir, exist_ok=True)
    with open(f'{stats_dir}/stats.txt', 'w') as file:
        file.write(content)

    print("Content has been written to stats.txt")
    
    ## Create train set
    
    """
    In this part we need to convert the chromosomal positions into sequences and we do that by extracting the sequences from the GRCh38 reference genome
    
    """
    
    # Path to your reference genome FASTA file
    reference_genome_file = '/mnt/raid1/thalassini/home/Downloads/Homo_sapiens.GRCh38.dna.toplevel.fa'
    #reference_genome_file = '/mnt/raid1/thalassini/home/Downloads/Homo_sapiens.GRCh38.dna_sm.primary_assembly.chrAdded.fa'
 


    # Load the reference genome using pyfaidx
    fasta = Fasta(reference_genome_file)
    # We need to update the first 25 fasta.keys in order to match the NCBI format, the rest will not be used
    
    print("updating fasta keys...")    
    updated_fasta = {}

    count = 0
    for k, v in fasta.items():
        if count < 25:
            updated_fasta[mapping_table.get(k, k)] = v
            count += 1
        else:
            updated_fasta[k] = v
            
            
    # Create true positive test
    print("Creating true positive set...")
    true_positives=[]
    # Loop through the DataFrame and retrieve values of 'Chromosome', 'Start', and 'End' columns for each row
    for index, row in chipseq_filtered_df.iterrows():
        chromosome = row['Chromosome']
        start = row['Start']
        end = row['End']
        true_positives.append(updated_fasta[chromosome][start-100:end+100].seq)
        
    print("Creating true negative set...")    
    true_negatives=[]
    # Loop through the DataFrame and retrieve values of 'Chromosome', 'Start', and 'End' columns for each row
    for index, row in negative_df.iterrows():
        chromosome = row['Chromosome']
        start = row['Start']
        end = row['End']
        true_negatives.append(updated_fasta[chromosome][start-100:end+100].seq)
        
    content+=f"We have {len(true_negatives)} negative sequences and {len(true_positives)} positive sequences\n"
    content+=f"True positives are {len(true_positives)/(len(true_negatives)+len(true_positives))*100}% of the dataset\n"
    content+=f"True negatives are {len(true_negatives)/(len(true_negatives)+len(true_positives))*100}% of the dataset\n"
    with open(f'{stats_dir}/stats.txt', 'w') as file:
        file.write(content)
    print("dataset created")
    
    
    ## Checking the length of the sequences
    print("Checking length of sequences...")
    # Check the length of each string in the list
    lengths_pos = [len(item) for item in true_positives]
    content+=f"Negative sequences Maximum Lenght:{max(lengths_pos)}, Μinimum Length:{min(lengths_pos)}\n"
    lengths_neg = [len(item) for item in true_negatives]
    content+=f"Positive sequences Maximum Lenght:{max(lengths_neg)}, Μinimum Length:{min(lengths_neg)}\n"
    with open('stats.txt', 'w') as file:
        file.write(content)
       

    
    # Combine the two sets in a df
    print("Combining the two sets in one csv..")
    # Create DataFrames for negative and positive sequences
    true_negatives_df = pd.DataFrame({'data': true_negatives, 'class': 0})
    true_positives_df= pd.DataFrame({'data': true_positives, 'class': 1})

    # Concatenate both DataFrames into a single DataFrame
    combined_df = pd.concat([true_negatives_df, true_positives_df], ignore_index=True)
    # Export the DataFrame to a CSV file
    # Join the home directory path with the new directories
     
    dataset_dir = os.path.join(home_dir,f"filtered_fimo/{tf}/datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    combined_df.to_csv(f'{dataset_dir}/data.csv', index=False)
    

 
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process FIMO data')
    parser.add_argument('--biomart_indir', help='Processed mart.txt directory path')
    parser.add_argument('--tf', help='Name of TF in upper case')

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.biomart_indir,args.tf)



    
    
    
   


    
    
    
    
    


# In[ ]:




