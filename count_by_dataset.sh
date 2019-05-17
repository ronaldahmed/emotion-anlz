#!/bin/bash

echo "Distribution per dataset"
echo "========================================================================"
cut -f 2 unified-dataset.tsv | sort | uniq -c | sort -rn
echo ""


for dataset in $(cut -f 2 unified-dataset.tsv | uniq); do
	echo "Label distribution: $dataset"
	echo "========================================================================"
	cat unified-dataset.tsv | grep "$dataset" | cut -f 4 | sort | uniq -c | sort -rn
	echo ""
done