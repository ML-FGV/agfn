library(pcalg) 

toMAG <- function(graphsDir) {
	filenames <- list.files(graphsDir, full.names = TRUE) 

	for (i in seq_along(filenames)) {
		pag.amat <- read.csv(filenames[i]) 
		pag.amat <- data.matrix(pag.amat) 
		mag.amat <- pag2magAM(pag.amat, 1) 
		write.csv(mag.amat, filenames[i], row.names = FALSE) 
	} 
} 

args <- commandArgs(trailingOnly = TRUE) 
toMAG(args[1]) 
