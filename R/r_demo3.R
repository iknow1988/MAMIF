setwd("/Users/ethanchen/Desktop/2019SEM1/DataProject/Shared_repo/MAMIF/")


library(lidR)
library(dplyr)


#### load and only use xyz coordinate
dat = readLAS(files = "./SP01.las" ,select = "xyz")

pre_processing <- function(las_df , boundary,shrunk_factor =0.05){
      
      dat <- las_df@data
      xy_filtered = dat %>% filter(dat$X > -boundary , dat$X < boundary,
                                   dat$Y > -boundary , dat$Y < boundary) 

      small_dat = sample_n(xy_filtered , nrow(xy_filtered) * shrunk_factor )
      
      return(LAS(small_dat))
}

small_dat <- pre_processing(las_df = dat ,boundary = 15  , shrunk_factor = 0.2)
#plot(small_dat)


col <- pastel.colors(200)

chm <- grid_canopy(small_dat, 0.5, p2r(0.3))
ker <- matrix(1,3,3)
chm <- raster::focal(chm, w = ker, fun = mean, na.rm = TRUE)

ttops <- tree_detection(chm, lmf(4, 2))
las   <- lastrees(small_dat, dalponte2016(chm, ttops))
plot(las,color = "treeID", colorPalette = col)


data = las@data
plot(LAS(subset(data, treeID==22)%>% select(1:3)))
length(unique(data[["treeID"]]))

