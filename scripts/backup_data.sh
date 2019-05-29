tdir=/share/PI/sabatti/feat_viz/real_analysis_result/analysis_050719
cdir=hepa_data
tfname=hepa_data.tar.gz

cd $tdir 
tar -zcvf $tfname $cdir
echo "created ${tfname}"
scp $tfname jjzhu@rice.stanford.edu:/home/jjzhu/afs-home/WWW/fileshare/aloe
