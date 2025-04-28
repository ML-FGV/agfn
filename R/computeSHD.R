rm(list=ls())
library(pcalg)
library(dagitty)
library(stringr)
library(jsonlite)
library(igraph)
source("sim_utils.R")

####################
# Helper Functions #
####################


# returns an pcalg amat (adjacency matrix) of type amat.pag (same type for MAGs),
# where:
# 0: No edge
# 1: Circle
# 2: Arrowhead
# 3: Tail
dagitty2amat <- function(adagg, type="mag") {
  e <- NULL
  edg <- dagitty::edges(adagg)
  node_names <- names(adagg)
  ans_mat <- matrix(
    data = 0, nrow = length(node_names),
    ncol = length(node_names),
    dimnames = list(node_names, node_names)
  )

  diredg <- subset(edg, e == "->")
  if (length(diredg) > 0) {
    ans_mat[as.matrix(diredg[c("w", "v")])] <- 3
    ans_mat[as.matrix(diredg[c("v", "w")])] <- 2
  }

  bidiredg <-  subset(edg, e == "<->")
  if (length(bidiredg) > 0) {
    ans_mat[as.matrix(bidiredg[c("w", "v")])] <- 2
    ans_mat[as.matrix(bidiredg[c("v", "w")])] <- 2
  }
  return(ans_mat)
}

#WARNING: magg does not necessarily have nodes ordered as in amat
getMAG <- function(amat, type="pag") {
  if (type == "pag") {
    amat.mag <- pcalg::pag2magAM(amat, 1)
    #plotAG(amat.mag)
    magg <- NULL
    if (!is.null(amat.mag)) {
      magg <- pcalg::pcalg2dagitty(amat.mag, labels=colnames(amat), type="mag")
      #plot(magg)
    }
  } else {
    adagg <- pcalg::pcalg2dagitty(amat, labels=colnames(amat), type="dag")
    magg <- dagitty::toMAG(adagg)
    amat.mag <- dagitty2amat(magg, type="mag")
  }

  # making sure that the order of the labels is preserved.
  amat.mag <- amat.mag[colnames(amat), colnames(amat)]

  return(list(amat.mag = amat.mag, magg=magg))
}



dagittyCIOracle <- function(x, y, S, suffStat) {
  g <- suffStat$g
  labels <- names(g)
  if (dagitty::dseparated(g, labels[x], labels[y], labels[S])) {
    return(1)
  } else {
    return(0)
  }
}


# receives a dagitty g of type "dag" or "mag" and returns
# the true PAG as an pcalg fci object
getTruePAG <- function(g, verbose = FALSE) {
  indepTest <- dagittyCIOracle
  if (dagitty::graphType(g) == "dag") {
    g <- dagitty::toMAG(g)
  }
  labels=names(g)
  suffStat <- list(g=g, labels=labels)
  truePag <- pcalg::fci(suffStat,
                        indepTest = indepTest,
                        labels = labels, alpha = 0.9999,
                        verbose = verbose)
  return(truePag)
}

shd_PAG <- function(amat.trueP, amat.estP, verbose=FALSE) {
  strDist <- 0
  amat.estP <- amat.estP[rownames(amat.trueP), colnames(amat.trueP)]
  if (!is.null(amat.trueP)) {
    if (any(dim(amat.estP) != dim(amat.trueP))) {
      stop("amat.trueP and amat.estP must have same dimensions.")
    }
    strDist <- length(which(amat.estP - amat.trueP != 0))
  }
  return(strDist)
}

##########################
# simTypes <- c("chain4", "iv", "collfork", "descColl", "discr1_nc")
#    chain4: X -> Z -> Y -> W
#    iv: A -> B -> C; B <- D -> C
#    descColl: D -> C <- E; A <- C -> B
#    collfork: X -> Z <- W -> Y; X <-> W; Z <-> Y
#    discr1_nc: A -> Y; X -> A <-> B -> Y
##############################
getTrueGraphs <- function(simType) {
  if (! (simType %in% c('ecoli', 'yeast'))) {
    adag_out <- getDAG(simType)
    #plot(as(adag_out$amat, "graphNEL"))
    trueMAG <- toMAG(adag_out$dagg)
  } else {
    trueMAG <- getMAG(simType)
  }

  amat.mag <-  dagitty2amat(trueMAG)
  #plotAG(amat.mag)
  truePAG <- getTruePAG(trueMAG)
  amat.pag <- truePAG@amat
  amat.pag <- amat.pag[colnames(amat.mag), colnames(amat.mag)]

  #plotAG(amat.pag)
  return(list(dagg_objs=list(mag = trueMAG, pag = truePAG),
              amat=list(mag=amat.mag, pag=amat.pag)))
}

getDagittyADMG <- function(biAdjM, dirAdjM) {
  admgStr <- "dag{"
  labels <- colnames(biAdjM)
  for(i in 1:length(labels)) {
    admgStr <- paste0(admgStr, labels[i], "\n")
  }
  bi_ids <- which(upper.tri(biAdjM) & biAdjM == 2, arr.ind = T)
  labels <- colnames(biAdjM)
  if (nrow(bi_ids) > 0) {
    for (i in 1:nrow(bi_ids)) {
      admgStr <- paste0(admgStr,
          paste0(labels[bi_ids[i, 1]], " <-> ",
                 labels[bi_ids[i, 2]]))
      if (i < nrow(bi_ids)) {
        admgStr <- paste0(admgStr, "; ")
      }
    }
  }
  dir_ids <- which(dirAdjM == 3, arr.ind = T)
  if (nrow(dir_ids) > 0) {
    if (nrow(bi_ids) > 0) {
      admgStr <- paste0(admgStr, "; ")
    }
    for (i in 1:nrow(dir_ids)) {
      if (dirAdjM[dir_ids[i, 2], dir_ids[i, 1]] == 2) {
        admgStr <- paste0(admgStr,
                          paste0(labels[dir_ids[i, 2]], " -> ",
                                 labels[dir_ids[i, 1]]))
        if (i < nrow(dir_ids)) {
          admgStr <- paste0(admgStr, "; ")
        }
      }
    }
  }
  admgStr <- paste0(admgStr, "}")
  return(dagitty(admgStr))
}

# amatType: indicates whether the adjacency matrices in amatPath are either a "mag" ou a "pag"
# shdType:
# if "pag", then compares the truePAG with a PAG corresponding to the MAGs in amatPath,
# otherwise, compares the the trueMAG with the MAGs in amatPath
computeSHD <- function(amatPath, simType, shdType="pag", amatType="mag", verbose = FALSE) {
  filenames <- list.files(amatPath, full.names = TRUE)
  results_df <- data.frame()
  truegraphs <- getTrueGraphs(simType)
  for (i in seq_along(filenames)) {
    if (shdType == "pag") {
      amat.true <- truegraphs$amat$pag
    } else {
      amat.true <- truegraphs$amat$mag
    }
    amat.est <- as.matrix(read.csv(filenames[i]))
	colnames(amat.est) <- colnames(amat.true)
    rownames(amat.est) <- colnames(amat.est)

    # making sure the the labels of the true and estimated graphs match
    if (! length(colnames(amat.est)) == length(colnames(amat.est)) ||
                 !all(colnames(amat.est) %in% colnames(amat.true))) {
      stop("Error: labels of the true and estimated graphs do not match. ")
    }
    amat.true <- amat.true[colnames(amat.est), colnames(amat.est)]

    if (shdType == "pag" && amatType == "mag") {
      # converts mag to a PAG
      dagg.est <- getMAG(amat.est, type ="mag")$magg # converts the mag adj matrix to a mag dagitty object
      amat.est <- getTruePAG(dagg.est)@amat # gets the adj matrix of the corresponding pag
    } else if (shdType == "mag" && amatType == "pag") {
      # converts pag into MAG
      amat.est <- getMAG(amat.est, type ="pag")$amat.mag # converts the pag adj matrix into the canonical mag adj matrix
    }
    amat.est <- amat.est[colnames(amat.true), colnames(amat.true)]

    amat.true_json <- toJSON(amat.true)
    amat.est_json <- toJSON(amat.est) #, pretty = TRUE)

    # Compute the SHD -- it works for both PAGs and MAGs
    cur_shd <- shd_PAG(amat.true, amat.est, verbose = verbose)


    print(amat.true)
    print(amat.est)


    # Write the metrics to a data-frame
    results_df[i, "dataset"] <- simType
    results_df[i, paste0("estimated_", shdType)] <- amat.est_json
    results_df[i, paste0("true_", shdType)] <- amat.true_json
    results_df[i, "SHD"] <- cur_shd
    results_df[i, "graph"] <- filenames[i]
    if (verbose & !(i %% 25)) {
     print(simType)
     print(i)
     print(results_df[i,])
    }
  }
  return(results_df)
}


computeSHD("../5mag_ds/5mag_ds_10feed", "sachs5", "mag","mag")

###############
# Testing SHD #
###############

runTestSachs <- function() {
  simType <- "sachs5"
  amatPath <- "./test/amat_sachs5/"
  dat <- read.csv(paste0("./test/datasets/", simType, ".csv"))

  # Run FCI

  indepTest <- gaussCItest
  suffStat <- list(C = cor(dat), n = nrow(dat))
  labels <- colnames(dat)

  fit_fci <- pcalg::fci(suffStat, indepTest = indepTest,
                        skel.method = "stable", labels = labels,
                        type = "normal", alpha = 0.05,
                        verbose = TRUE, conservative = FALSE,
                        maj.rule = FALSE)

  fci_amat.pag <- fit_fci@amat # adj matrix of the PAG
  # plotAG(fci_amat.pag)

  # saving a pag adj matrix to the file
  write.csv(fci_amat.pag,
            file = paste0(amatPath, "fci_amat_", simType, ".csv"),
            row.names = FALSE)

  cat("Comparing FCI PAG and true PAG, given FCI PAG:")
  computeSHD(amatPath, simType, shdType = "pag", amatType = "pag", verbose=TRUE)

  cat("Comparing FCI MAG and true MAG, given FCI PAG:")
  computeSHD(amatPath, simType, shdType = "mag", amatType = "pag", verbose=TRUE)


  # Getting a MAG in the MEC
  fci_amat.mag <- getMAG(fci_amat.pag, type="pag")$amat.mag
  # plotAG(fci_amat.mag)


  # saving a mag adj matrix to the file
  write.csv(fci_amat.mag,
            file = paste0(amatPath, "fci_amat_", simType, ".csv"),
            row.names = FALSE)

  cat("Comparing FCI PAG and true PAG, given FCI MAG:")
  computeSHD(amatPath, simType, shdType = "pag", amatType = "mag", verbose=TRUE)


  cat("Comparing FCI MAG and true MAG, given FCI MAG:")
  computeSHD(amatPath, simType, shdType = "mag", amatType = "mag", verbose=TRUE)

}




if (getOption('run.computeSHD', default = TRUE)) {
  args <- commandArgs(trailingOnly = TRUE)
  amatPath <- args[1]
  simType <- args[2]
  results_df <- computeSHD(amatPath, simType)
  write.csv(results_df, paste0(amatPath, '_results_samplings.csv'), row.names = FALSE)
}

