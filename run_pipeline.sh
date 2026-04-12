#!/bin/bash

set -e

echo " Starting NLP Plagiarism Pipeline... "

echo -e "\n[Phase 1] Executing representation scripts..."

for rep_script in representation/*.py; do
    if [ -f "$rep_script" ]; then
        echo " -> Running $rep_script"
        python3 "$rep_script"
    fi
done

echo -e "\n[Phase 1] Complete! Data representations generated."

echo -e "\n[Phase 2] Executing classifier scripts..."

csv_filenames=()
for file in data/*.csv; do
    csv_filenames+=("$(basename "$file")") 
done

echo "CSV files to process: ${csv_filenames[*]}"
echo "----------------------------------------"

for clf_script in classifier/*.py; do
    if [ -f "$clf_script" ]; then
        echo " -> Running $clf_script"
        
        python3 "$clf_script" "${csv_filenames[@]}"
    fi
done

echo " Pipeline execution finished successfully! "
echo " Check your 'resultats/' folder for output."
python3 resultats/results.py