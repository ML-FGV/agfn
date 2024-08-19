library(igraph)
library(SEMgraph)
library(lavaan)

computeBIC <- function(mags, dat, group_var=NULL) {

  getBIC <- function(ig.mag, dat, group_var=NULL) {
    fit <- tryCatch({
      if (!is.null(group_var)) {
        SEMgraph::SEMrun(graph=ig.mag, data=dat, group=dat[,group_var]) #, algo = "ricf")
      } else {
        SEMgraph::SEMrun(graph=ig.mag, data=dat, group=NULL) #, algo = "ricf")
      }
    }, error = function(cond) {
      message("Error computing BIC")
      message(conditionMessage(cond))
      NA
    })
    
    if (length(fit) == 1 && is.na(fit)) {
      return(NA)
    } else {
      summ_fit <- lavaan::summary(fit$fit, fit.measures = TRUE)
      return(summ_fit$fit["bic2"]) # bic2 vs bic
    }
  }
  
  update_amat_colnames <- function(mag) {
    colnames(mag) <- colnames(dat) 
    mag
  } 

  mags <- lapply(mags, update_amat_colnames) 
  imags <- lapply(mags, igraph::graph_from_adjacency_matrix) 
  bics <- sapply(imags, getBIC, dat=dat, group_var=group_var)
  return(bics)
}

############
# Example: #
############

runExample = FALSE
if (runExample) {

  library(dagitty)
  
  node_labels <- c("A", "B", "C", "D")
  
  # B <- A -> C; A -> D <- C
  mag1 <- matrix(0, 4, 4)
  mag1[1,3] = 1; mag1[3,1] = 0 # 1 -> 3
  mag1[1,2] = 1; mag1[2,1] = 0 # 1 -> 2
  mag1[2,4] = 1; mag1[4,2] = 0; # 2 -> 4
  mag1[3,4] = 1; mag1[4,3] = 0; # 3 -> 4
  colnames(mag1) <- node_labels
  
  # B -> A -> C; A -> D <- C
  mag2 <- matrix(0, 4, 4)
  mag2[1,3] = 1; mag2[3,1] = 0 # 1 -> 3
  mag2[2,1] = 1; mag2[1,2] = 0 # 2 -> 1
  mag2[2,4] = 1; mag2[4,2] = 0; # 1 -> 4
  mag2[3,4] = 1; mag2[4,3] = 0; # 3 -> 4
  colnames(mag2) <- node_labels
  
  # B <-> A -> C; A -> D <- C
  mag3 <- matrix(0, 4, 4)
  mag3[1,3] = 1; mag3[3,1] = 0 # 1 -> 3
  mag3[2,1] = 1; mag3[1,2] = 1 # 2 <-> 1
  mag3[2,4] = 1; mag3[4,2] = 0; # 2 -> 4
  mag3[3,4] = 1; mag3[4,3] = 0; # 3 -> 4
  colnames(mag3) <- node_labels
  
  # A B C D 
  mag4 <- matrix(0, 4, 4)
  mag4[1,3] = 0; mag4[3,1] = 0 
  mag4[2,1] = 0; mag4[1,2] = 0 
  mag4[2,4] = 0; mag4[4,2] = 0; 
  mag4[3,4] = 0; mag4[4,3] = 0; 
  colnames(mag4) <- node_labels
  
  mag5 <- matrix(c(0, 1, 1, 1,
                0, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0), nrow = 4, byrow = TRUE) 

  mags <- list(mag1, mag2, mag3, mag4, mag5)
  
  plot.igraph(igraph::graph_from_adjacency_matrix(mag5))
  
  test_data <- dagitty::simulateSEM(graph2dagitty(mag3, graphType = "dag"), N=1000)
  
  test_data$D[which(test_data$D <= 0)] = 0
  test_data$D[which(test_data$D > 0)] = 1

  # Use local data 
  # test_data <- read.csv('data/chain4_dat01.csv') 
  # results: 
  #     bic2     bic2     bic2     bic2          
  # 3798.114 3798.114 3798.114 3822.137       NA 

  # fit <- lm(C ~ A + B, data=test_data)
  # summary(fit)
  # 
  # fit <- glm(D ~ B + C + A, data=test_data, family = "binomial")
  # summary(fit)
  
  bics <- computeBIC(mags, test_data, group_var = "D")
  print(bics)
}
