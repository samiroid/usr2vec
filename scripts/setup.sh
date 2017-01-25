sma_toolkit="/home/ubuntu/efs/work/projects/sma_toolkit/"
embeddings_txt="/home/ubuntu/efs/work/embeddings/mental_health/w2v/embs_emoji_2_400.txt"
embeddings_bin="/home/ubuntu/efs/work/embeddings/mental_health/w2v/bin/"
#this is where the input files will be stored (or linked to)
mkdir raw_data 
#DATA folder
rm -rf DATA
mkdir DATA
mkdir DATA/embeddings
rm -rf code/sma_toolkit
#link toolkit and embeddings
ln -s ${sma_toolkit} code/sma_toolkit
ln -s $embeddings_txt DATA/embeddings/word_embeddings.txt
ln -s $embeddings_bin DATA/embeddings/bin

#preprocess corpus
#mode="SMALL"
#python scripts/preprocess.py raw_data/user_corpus.txt DATA/txt/mental_health_corpus.txt $mode
mkdir DATA/txt
ln -s /home/ubuntu/efs/work/projects/usr2vec/raw_data/user_corpus.txt DATA/txt/mental_health_corpus.txt