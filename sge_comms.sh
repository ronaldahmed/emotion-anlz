qsub -pe smp 30 -cwd -l mem_free=10G,act_mem_free=10G,h_data=15G -p -50 \
-o rf.out \
-e rf.err \
run_classifier.sh rf 30


qsub -pe smp 30 -cwd -l mem_free=10G,act_mem_free=10G,h_data=15G -p -50 \
-o knn.out \
-e knn.err \
run_classifier.sh knn 30