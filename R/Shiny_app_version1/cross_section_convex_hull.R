
setwd("/Users/ethanchen/Desktop/2019SEM1/DataProject/MAMIF/R/Shiny_app_version1")
library(lidR)
library(dplyr)
library(rgl)
# To plot raster layer in 3D.
library(rasterVis)
library(TreeLS)
library(styler)
library(rLiDAR)
source("../utils.R")


###########################
# Data preprocessing
###########################
dataPath <- "/Users/ethanchen/Desktop/2019SEM1/DataProject/Shared_repo/"

file_name <- 'SP03_Stony.las'
file_path <- paste(dataPath ,file_name , sep = '')

dat <- lidR::readLAS(file_path)

small_dat <- pre_processing(las_df = dat, boundary =  40, shrunk_factor = 0.15)

small_dat_with_ground <- lasground(small_dat, csf())

plot(small_dat_with_ground)




small_dat_with_ground@data[,1:3]

dtm <- grid_terrain(small_dat_with_ground ,algorithm = knnidw(k = 6L, p = 2))

small_dat_with_ground <- small_dat_with_ground - dtm
small_dat_no_ground <- filter_class(small_dat_with_ground , class_index = 1)



############################
# Variable 1 : Tree Segmentation
############################

las <- tree_segmentation(small_dat_no_ground,   algorithm = 2)
metric<- tree_metrics(las,.stdtreemetrics)
metric
hulls<- tree_hulls(las)
hulls@data<- dplyr::left_join(hulls@data, metric@data)

head(hulls@data)

#####################
# after fileter the point with a given height, 2D concex hull
#####################
summary(las@data$Z)
hist(las@data$Z)
tmp <- las@data %>% filter( Z > 5 & Z < 7)
dim(tmp)
tmp <- LAS(tmp)
tmp
concave_hulls = tree_hulls(tmp, "convex")
sp::plot(concave_hulls)

dim(concave_hulls@data)

length(concave_hulls@polygons)

concave_hulls@polygons[3][[1]]@Polygons

concave_hulls

library(maptools)

writeSpatialShape(concave_hulls , 'test_concave_hulls')
##########################
# 3D hull
#####################

# Set the alpha
alpha<-0.6

# Set the plotCAS parameter
plotit=TRUE

# Set the convex hull color
col="forestgreen"

xyzid <- tmp@data[,1:3]
xyzid$id <- tmp@data$treeID

# Get the volume and surface area
library(rgl)
open3d() 
volumeList<-chullLiDAR3D(xyzid=xyzid, plotit=plotit, col=col,alpha=alpha)
summary(volumeList) # summary



plot3d(xyzid[,1:3], add=TRUE)   # add the 3D point cloud
axes3d(c("x+", "y-", "z-"))                 # axes
grid3d(side=c('x+', 'y-', 'z'), col="gray") # grid
title3d(xlab = "UTM Easthing", ylab = "UTM Northing",zlab = "Height", col="red")


volumeList

