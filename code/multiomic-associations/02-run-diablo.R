
library(tidyverse)
library(mixOmics)
library(igraph)

X1 <- read.csv('../../results/multiomic-associations/diablo-setup/microbes.csv', row.names=1)
X2 <-  read.csv('../../results/multiomic-associations/diablo-setup/luminex.csv',row.names=1)
X3 <-  read.csv('../../results/multiomic-associations/diablo-setup/clinical.csv',row.names=1)

drops <- c('V1AF14')
X3 <- X3[ , !(names(X3) %in% drops)]

Y_pec <-  read.csv('../../results/multiomic-associations/diablo-setup/metadata.csv', row.names=1) %>% 
              mutate(PEC_case=PEC_case=='True')
row.names(X1)=row.names(X2)
row.names(X3)=row.names(X2)
row.names(Y_pec)=row.names(X2)

group_list <- list(row.names(Y_pec), 
                   which(Y_pec$BMI>=25),
                   which(Y_pec$BMI<25)
                   )

save_dir_group <- c('all-cohort',
                    'bmi-greater25', 
                    'bmi-less25')

for(i in 1:3){
  
  set.seed(123)
  
  inds <- group_list[[i]]

  
  
  X <- list(micro = X1[inds,] %>% as.matrix, 
            luminex = X2[inds,] %>% as.matrix, 
            clinical=X3[inds,] %>% as.matrix
            )
  Y <- Y_pec[inds,]$PEC_case
  
  # for square matrix filled with 0.1s
  design = matrix(0.1, 
                  ncol = length(X),
                  nrow = length(X), 
                  dimnames = list(names(X), 
                                  names(X))
                  )
  diag(design) = 0 # set diagonal to 0s
  
  basic.diablo.model = block.splsda(X = X, 
                                    Y = Y, 
                                    ncomp = 5,
                                    design = design) 
  
  # run component number tuning with repeated CV
  perf.diablo = perf(basic.diablo.model, 
                     validation = 'Mfold', 
                     folds = 10, 
                     nrepeat = 10 ) 
  # plot(perf.diablo) # plot output of tuning
  
  ncomp = perf.diablo$choice.ncomp$WeightedVote["Overall.BER", "centroids.dist"] 
  
  test.keepX = list (micro = c(2,5,10), 
                     luminex = c(2,5,10),
                     clinical = c(2,5,10) 
                     )
  
  # run the feature selection tuning
  tune.TCGA = tune.block.splsda(X = X, 
                                Y = Y, 
                                ncomp = ncomp, 
                                test.keepX = test.keepX, 
                                design = design,
                                validation = 'Mfold', 
                                folds = 5, 
                                nrepeat = 1,
                                dist = "centroids.dist"
                                )
  
  list.keepX = tune.TCGA$choice.keepX
  
  # set the optimised DIABLO model
  final.diablo.model = block.splsda(X = X, 
                                    Y = Y, 
                                    ncomp = ncomp, 
                                    keepX = list.keepX, 
                                    design = design )
  
  plotDiablo(final.diablo.model, ncomp = 1)
  plotLoadings(final.diablo.model,
               block=c(1,2,3),
               comp = 1, 
               col.ties='grey',
               contrib = 'max', 
               method = 'median', 
               )
  
  
  dev.print(pdf, 
            file=paste0("../../results/multiomic-associations/",
                            save_dir_group[i],
                            "/diablo-loadings.pdf")
            )
  
  do.call(rbind, unname(final.diablo.model$loadings)) %>% 
    as.data.frame() %>% filter( comp1!=0 ) %>% 
    write.csv(
      paste0("../../results/multiomic-associations/",
             save_dir_group[i],
             "/diablo-loadings.csv")
    )

  set.seed(123)
  q <- network(final.diablo.model,
          blocks = c(1,2,3),
          color.node = c('darkorchid', 'brown1', 'lightgreen'),
          cutoff = 0.5,
          lwd.edge = 10,
          block.var.names=F,
          alpha.node=c(0.5),
          cex.node.name=0.00001,
          save='pdf',
       )
  
  
  q$gR %>% 
    as_long_data_frame() %>% 
    write.csv(
        paste0("../../results/multiomic-associations/",
               save_dir_group[i],
               "/diablo-graph.csv")
              )
  


  set.seed(123)
   network(final.diablo.model,
           blocks = c(1,2,3),
           color.node = c('darkorchid', 'brown1', 'lightgreen'),
           cutoff = 0.5,
           lwd.edge = 10,
           alpha.node=c(0.5),
           save='pdf',
           name.save=paste0("../../results/multiomic-associations/",
                            save_dir_group[i],
                            "/diablo-network")
          )
}



