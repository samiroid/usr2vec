#!/bin/bash
out="DATA/out/usr2vec_400_master.txt"
first=true
for f in DATA/out/*.txt; do 	
#	cat $f
	if [[ ${first} = true ]]; then		
		echo "merging: " $f;
		cat $f > ${out}; 
		first=false
		continue
	fi
	if [[ ${f} != *"master"* ]]; then    
		echo "merging: " $f;
		sed '1d' $f >> ${out}; 
	fi	
done
echo "all merged into " ${out}


