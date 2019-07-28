setwd("/Users/ethanchen/Desktop/2019SEM1/DataProject/Shared_repo/MAMIF/")



pre_processing <- function(las_df , boundary,shrunk_factor =0.05){
      
      dat <- las_df@data
      # remove point that are way too high than all other points.
      Z_upperbound = quantile(x = dat$Z , prob = 0.999999)
      
      xy_filtered = dat %>% filter(dat$X > -boundary , dat$X < boundary,
                                   dat$Y > -boundary , dat$Y < boundary , dat$Z < Z_upperbound) 
      
      small_dat = sample_n(xy_filtered , nrow(xy_filtered) * shrunk_factor )
      
      return(LAS(small_dat))
}





filter_class <- function(las_df , class_index){
      tmp_df <- las_df@data
      tmp_df <- tmp_df %>% filter(tmp_df$Classification == class_index)
      
      return(LAS(tmp_df))
}

compute_grid_canopy <- function(las_df,radius_length, resolution = 0.5 , smooth= TRUE){
   
   chm <- grid_canopy(las = las_df , res = resolution , p2r(radius_length))
   
   if (smooth) {
      # Smooth the curve 
      ker <- matrix(1,3,3)
      # Calculate focal ("moving window") values for the neighborhood of focal cells using a matrix of weights, perhaps in combination with a function.
      
      chm <- raster::focal(chm, w = ker, fun = mean, na.rm = TRUE)
   }
   
   return(chm)
}



tree_segmentation <- function(las_df, p2r_radius_length=0.2, lmf_ws=4, algorithm =1,resolution = 0.5 , smooth= TRUE){
   
   chm <- compute_grid_canopy(las_df = las_df , radius_length = p2r_radius_length ,resolution = resolution , smooth = smooth)
   
   ttops <- tree_detection(chm, lmf(ws=lmf_ws, hmin=2))
   print(length(ttops$treeID))
   
   
   if (algorithm == 1){
      las   <- lastrees(las_df, silva2016(chm, ttops))
   }else if(algorithm == 2){
      las   <- lastrees(las_df, dalponte2016(chm, ttops))
   }
   n_tree <- length(na.omit(unique(las@data$treeID)))
   
   
   print(paste("Number of tree : " ,n_tree ))
   
   return(las)
}