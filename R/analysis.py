import numpy as np 
import pandas as pd 

DATASETS = ['1be', '4be', 'discr1_c']
INDICES = list(range(3, 6)) 
SAMPLE_DAG = [True, False] 

RESULTS_DIR = '../results'

def join_results(dataset, sample_dag, idx): 
    filestem = f'{dataset}_{idx:02d}_dag_{sample_dag}_normalized_False' 
    shd_path = f'{RESULTS_DIR}/{filestem}_results_samplings.csv' 
    bic_path = f'{RESULTS_DIR}/results_from_{filestem}.csv'  

    shd = pd.read_csv(shd_path) 
    shd.set_index('graph', inplace=True) 
    bic = pd.read_csv(bic_path) 
    bic.set_index('filenames', inplace=True)

    merged = shd.join(bic)
    filt = merged.bic >= np.sort(merged.bic)[-5]     
    return merged.SHD[filt].mean(), merged.bic[filt].mean() 

def main(): 

    for dataset in DATASETS: 
        for sample_dag in [True, False]:
            results_for_method = list()  
            results_for_method_bic = list() 
            for idx in INDICES: 
                shd, bic = join_results(dataset, sample_dag, idx) 
                results_for_method.append(shd) 
                results_for_method_bic.append(bic)  
            print(dataset, sample_dag) 
            results_for_method = np.array(results_for_method) 
            results_for_method_bic = np.array(results_for_method_bic) 
            print(np.mean(results_for_method), results_for_method.std()) 
            print(np.mean(results_for_method_bic), results_for_method_bic.std()) 

if __name__ == '__main__': 
    main() 