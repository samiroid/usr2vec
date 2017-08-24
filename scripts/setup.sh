sma_toolkit="/home/ubuntu/efs/work/projects/sma_toolkit/"
embeddings_txt="/home/ubuntu/efs/work/projects/twitter_mh/DATA/pkl/tmh_jointvecs.txt"
embeddings_txt_dom="/home/ubuntu/efs/work/projects/twitter_mh/DATA/pkl/tmh_jointvecs_domain.txt"
# embeddings_bin="/home/ubuntu/efs/work/embeddings/mental_health/w2v/bin/"

#sma_toolkit="/Users/samir/Dev/projects/sma_toolkit/"
#embeddings_txt="/Users/samir/Dev/resources/embeddings/mental_health/w2v/embs_emoji_2_400.txt"
#embeddings_bin="/Users/samir/Dev/resources/embeddings/mental_health/w2v/bin/"

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
ln -s $embeddings_txt_dom DATA/embeddings/word_embeddings_domain.txt

