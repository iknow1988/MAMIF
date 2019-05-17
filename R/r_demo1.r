setwd("/Users/ethanchen/Desktop/2019SEM1/DataProject/Shared_repo/MAMIF/")

library(lidR)
library(dplyr)


#### load and only use xyz coordinate
dat = readLAS(files = "./85_Seaview.las" ,select = "xyz")

nrow(dat@data)
plot(dat)

#### Filter the data
#### 
shrunk_factor = 0.05
small_dat = sample_n(dat@data , nrow(dat@data) * shrunk_factor )

#head(small_dat)

xy_filtered = small_dat %>% filter(small_dat$X > -10 , small_dat$X < 10,
                     small_dat$Y > -10 , small_dat$Y < 10) 

#head(xy_filtered)
#### contruct an LAS object for ploting 
xy_filtered <- LAS(xy_filtered)

plot(xy_filtered)
col <- pastel.colors(200)
# Using Li et al. (2012)
# adding treeID to the newly constructed LAS object
small_dat <- lastrees(xy_filtered, li2012(R = 3, speed_up = 2))

plot(small_dat, color = "treeID", colorPalette = col)

