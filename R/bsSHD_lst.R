rm(list=ls()) 
library(pcalg) 
library(dagitty)
library(stringr)
library(jsonlite)
library(igraph) 
source('sim_utils.R') 
options(run.computeSHD=FALSE) 
source('computeSHD.R') 

bsSHD <- function(filepath, numBS, simType, verbose = TRUE) {
    filenames <- list.files(filepath, full.names = TRUE) 
    results_df <- data.frame(
        graph = character(), 
        SHD = numeric(), 
        dataset = character(), 
        bs_idx = integer(), 
        mag = character(), 
        estimated_PAG = character() 
    )
    truegraphs <- getTrueGraphs(simType) 
    amat.truePAG <- truegraphs$amat$pag 
    
    for (i in seq(1, 10)) { # Number of datasets 
        for (j in seq(1, numBS)) {
            idx <- (i - 1) * numBS + j 
            amat.estMAG <- read.csv(filenames[idx]) 
            dagg.estMAG <- pcalg2dagitty(amat.estMAG, labels = colnames(amat.estMAG), type = "mag")
            amat.estPAG <- getTruePAG(dagg.estMAG)@amat
            estSHD <- shd_PAG(amat.truePAG, amat.estPAG) 
            amat.estPAG_json <- toJSON(amat.estPAG)
            amat.estMAG_json <- toJSON(amat.estMAG) 
            results_df[idx, "bs_idx"] <- j 
            results_df[idx, "SHD"] <- estSHD 
            results_df[idx, "dataset"] <- simType 
            results_df[idx, "graph"] <- filenames[idx] 
	        results_df[idx, "mag"] <- amat.estMAG_json 
            results_df[idx, "estimated_PAG"] <- amat.estPAG_json 
            if (verbose & !(idx %% 25)) {
                print(results_df[idx,]) 
            }
        }
    }
    return(results_df) 
} 

args <- commandArgs(trailingOnly = TRUE) 
method <- args[1] 
numBS <- 100
dataset <- args[2] 

filepath <- file.path('bs', dataset, method)
filepath_graphs <- file.path(filepath, 'graphs') 

results_df <- bsSHD(filepath_graphs, numBS, dataset) 

write.csv(results_df, paste0(filepath, '_results_sampling_bs.csv'), row.names = FALSE)

