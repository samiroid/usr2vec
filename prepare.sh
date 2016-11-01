utils="/home/ubuntu/work/projects/utils/my_utils"
embeddings="/home/ubuntu/work/embeddings"
mkdir DATA/embeddings
mkdir DATA/tmp
mkdir DATA/out
rm code/my_utils
ln -s ${utils} code/my_utils
ln -s ${embeddings} DATA/embeddings
