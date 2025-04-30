for dataset in 1be 4be discr1_c; do 
	for idx in {1..5}; do 
		for is_dag in True False; do 
			amatPath=results/${dataset}_0${idx}_dag_${is_dag}_normalized_False 
			Rscript computeSHD.R $amatPath $dataset 
			done 
	done 
done 
