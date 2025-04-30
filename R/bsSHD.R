rm(list=ls()) 
library(pcalg) 
library(dagitty)
library(stringr)
library(jsonlite)
library(igraph) 
source('sim_utils.R') 
options(run.computeSHD=FALSE) 
source('computeSHD.R') 

getPAG <- function(data, verbose = FALSE) {
    suffStat <- list(C = cor(data), n = nrow(data))  
    estPAG <- fci(suffStat, 
                indepTest = gaussCItest, 
                alpha = .05, 
		labels = names(data), 
                verbose = verbose)
    return(estPAG) 
}

bsSHD <- function(filepath, numBS, simType, verbose = TRUE) {
    filenames <- list.files(filepath, full.names = TRUE) 
    results_df <- data.frame(
        graph = character(), 
        SHD = numeric(), 
        dataset = character(), 
        bs_idx = integer(), 
        mag = character() 	
    )
    for (i in seq_along(filenames)) {
        # read the data 
        data <- read.csv(filenames[i]) 
        truegraphs <- getTrueGraphs(simType) 
        amat.truePAG <- truegraphs$amat$pag 
        # estimate the PAG 
        # compute the SHD for each boostrapped data set 
        for (j in seq(1, numBS)) {
            subsample <- sample(1:nrow(data), nrow(data), replace = TRUE) 
            amat.estPAG <- getPAG(data[subsample,])@amat
	        amat.estMAG <- toJSON(pag2magAM(amat.estPAG, sample(1:length(data), 1)))  
            estSHD <- shd_PAG(amat.truePAG, amat.estPAG) 
            idx <- (i - 1) * numBS + j 
            results_df[idx, "bs_idx"] <- j 
            results_df[idx, "SHD"] <- estSHD 
            results_df[idx, "dataset"] <- simType 
            results_df[idx, "graph"] <- filenames[i] 
	        results_df[idx, "mag"] <- amat.estMAG 
            if (verbose & !(idx %% 25)) {
                print(results_df[idx,]) 
            }
        }
    }
    return(results_df) 
} 

args <- commandArgs(trailingOnly = TRUE) 
filepath <- args[1] 
numBS <- 100
dataset <- args[2] 

results_df <- bsSHD(filepath, numBS, dataset) 

write.csv(results_df, paste0(dataset, '_results_sampling.csv'), row.names = FALSE)
