#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <filename.py> <number of copies>"
    exit 1
fi

FILENAME=$1
COPIES=$2

# Extract the directory path and the base name of the file
DIRPATH=$(dirname "$FILENAME")
BASENAME=$(basename "$FILENAME" .py)

# Loop to create copies in the same directory as the original file
for ((i=1; i<=COPIES; i++))
do
    cp "$FILENAME" "${DIRPATH}/temp${i}_${BASENAME}.py"
done
