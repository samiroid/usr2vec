#!/bin/sh
out="DATA/out/usr2vec_400_master.txt"
echo "" > ${out}
for f in DATA/out/*.txt; do 	
	if [[ ${f} != *"master"* ]]; then    
		echo "merging: " $f;
		sed '1d' $f >> ${out}; 
	fi	
done
echo "all merged into " ${out}


