setwd("/Users/ethanchen/Desktop/2019SEM1/DataProject/Shared_repo/MAMIF/")



pre_processing <- function(las_df , boundary,shrunk_factor =0.05){
      
      dat <- las_df@data
      # remove point that are way too high than all other points.
      Z_upperbound = quantile(x = dat$Z , prob = 0.9999)
      
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