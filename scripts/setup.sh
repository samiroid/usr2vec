# sma_toolkit="/home/ubuntu/efs/work/projects/sma_toolkit/"

sma_toolkit="/Users/samir/Dev/projects/sma_toolkit/"

#this is where the input files will be stored (or linked to)
#DATA folder
rm -rf DATA
mkdir DATA
rm -rf code/sma_toolkit
#link toolkit and embeddings
ln -s ${sma_toolkit} code/sma_toolkit



