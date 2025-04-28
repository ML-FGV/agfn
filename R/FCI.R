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

bsSHD <- function(filepath, simType, verbose = TRUE) {
    filenames <- list.files(filepath, full.names = TRUE) 
    results_df <- data.frame(
        graph = character(), 
        SHD = numeric(), 
        dataset = character(), 
        mag = character() 	
    )
    for (i in seq_along(filenames)) {
        # read the data 
        data <- read.csv(filenames[i]) 
        truegraphs <- getTrueGraphs(simType) 
        amat.truePAG <- truegraphs$amat$pag 
        # estimate the PAG 
        # compute the SHD for each boostrapped data set 
        amat.estPAG <- getPAG(data)@amat
	    amat.estMAG <- toJSON(pag2magAM(amat.estPAG, sample(1:length(data), 1)))  
        estSHD <- shd_PAG(amat.truePAG, amat.estPAG) 
        results_df[i, "SHD"] <- estSHD 
        results_df[i, "dataset"] <- simType 
        results_df[i, "data"] <- filenames[i] 
        results_df[i, "mag"] <- amat.estMAG 
        if (verbose & !(i %% 25)) {
            print(results_df[i,]) 
        }
    }
    return(results_df) 
} 

args <- commandArgs(trailingOnly = TRUE) 
dataset <- args[1] 
numBS <- 100

filepath <- file.path('fci', dataset) 
results_df <- bsSHD(filepath, dataset) 

write.csv(results_df, paste0(filepath, '_results_sampling.csv'), row.names = FALSE)

