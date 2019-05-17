setwd("/Users/ethanchen/Desktop/2019SEM1/DataProject/Shared_repo/MAMIF/")


pre_processing <- function(las_df , shruck_factor=0.05,boundary){
      xy_filtered = dat %>% filter(dat$X > -10 , dat$X < 10,
                                   dat$Y > -10 , dat$Y < 10) 
      
      
      small_dat = sample_n(xy_filtered , nrow(xy_filtered) * shrunk_factor )
      
      return(LAS(small_dat))
}
