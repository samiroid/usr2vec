server="ubuntu@ec2-54-84-37-190.compute-1.amazonaws.com"
server_path="/home/ubuntu/efs/work/projects/usr2vec/DATA/out/*"
rsync --rsh "ssh -i /Users/samir/Workspace/aws-rr.pem" -av ${server}:${server_path} DATA/out/
