#!/bin/bash

#Run this in order to convert the mart_export.tsv in bed format
#python convert_biomart.py --indir /mnt/raid1/thalassini/home/run_filter_fimo --outdir /mnt/raid1/thalassini/home/run_filter_fimo 

# Start time
start=$(date +%s)

'''
# List of TFs
tfs=('CTCF' 'JUN' 'NR3C1' 'TEAD4' 'ZNF274' 'USF2' 'TCF12' 'SRF' 'NRF1' 'MAZ' 'HES2' 'FOS' 'CEBPA' 'FOXA1' 'USF1' 'RELA' 'GABPA' 'ELF1' 'EGR1' 'ATF3' 'ZBTB33' 'SP1' 'CREB1' 'YY1' 'JUND' 'FOSL2' 'MAX' 'MYC' 'CEBPB' 'REST')
'''

# Read the contents of target_names_{}.txt into a Bash array
# Here we can select which of the tf lists we want to use and upload the corresponidng .txt
mapfile -t tfs < "/mnt/raid1/thalassini/home/Downloads/target_names_6.txt"

# Path to the Python script

python_script='/mnt/raid1/thalassini/home/Downloads/filter_fimo.py'

# Loop through each TF and run the command
for tf in "${tfs[@]}"; do
   
    python "$python_script" --biomart_indir /mnt/raid1/thalassini/home --tf "$tf"
done

# End time
end=$(date +%s)

# Calculate elapsed time
duration=$((end - start))
echo "Execution time: $duration seconds"
