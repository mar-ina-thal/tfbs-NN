
#Run this in order to convert the mart_export.tsv in bed format
#python convert_biomart.py --indir /mnt/raid1/thalassini/home/run_filter_fimo --outdir /mnt/raid1/thalassini/home/run_filter_fimo 

# Start time
start=$(date +%s)



# Read the contents of target_names_{}.txt into a Bash array
# Here we can select which of the tf lists we want to use and upload the corresponidng .txt
mapfile -t tfs < "/mnt/raid1/thalassini/home/Downloads/target_names_6.txt"

# Path to the Python script.py (CNN1.py or CNN2.py)


python_script='/mnt/raid1/thalassini/home/Downloads/CNN1.py'

# Loop through each TF and run the command
for tf in "${tfs[@]}"; do
   
    python "$python_script"  "$tf"
done

# End time
end=$(date +%s)

# Calculate elapsed time
duration=$((end - start))
echo "Execution time: $duration seconds"
