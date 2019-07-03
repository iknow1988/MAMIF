
setwd("/Users/ethanchen/Desktop/2019SEM1/DataProject/Shared_repo/MAMIF/")

source('utils.R')
dat = lidR::readLAS(files = "../SP03_Stony.las" ,select = "xyz")
small_dat<- pre_processing(las_df = dat ,boundary = 10  , shrunk_factor = 0.1)

small_dat_with_ground <- lasground(small_dat, csf())
dtm = grid_terrain(small_dat_with_ground, algorithm = knnidw(k = 6L, p = 2))
small_dat_with_ground <- small_dat_with_ground - dtm
lasData = filter_class(small_dat_with_ground , 1)

plot(lasData)

library(itcSegment)


##Creation of the look-up-table

lut<-matrix(6,2,data=NA)
lut<-data.frame(lut)
names(lut)<-c("H","CD")



lut$H<-c(2,10,15,20,25,30)
lut$CD<-c(0.5,1,2,3,4,5)

## function takes a while to run
se1<-itcLiDARallo(lasData$X,lasData$Y,lasData$Z,epsg=32632,lut=lut)
summary(se1)
plot(se1,axes=T)

## If we want to seperate the height of the trees by grayscales:

plot(se,col=gray((max(se$Height_m)-se$Height_m)/(max(se$Height_m)-min(se$Height_m))),axes=T)

## to save the data use rgdal function called writeOGR. For more help see rgdal package.




##epsg The EPSG code of the reference system of the X,Y coordinates.
se<-itcLiDAR(lasData$X,lasData$Y,lasData$Z,epsg=32632)
summary(se)
plot(se,axes=T)
## If we want to seperate the height of the trees by grayscales: 
plot(se,col=gray((max(se$Height_m)-se$Height_m)/(max(se$Height_m)-min(se$Height_m))),axes=T)


se@data
#Computation of the crown diameter from the crown area
se$CD_m<-2*sqrt(se$CA_m2/pi)
#DBH prediction
se$dbh<-NA
#The DBH value in centimeters.
se$dbh<-dbh(se$Height_m,se$CD_m,biome=0)
se@data

plot(x=se)

radius_length = 0.2
chm <- grid_canopy(small_dat_with_ground ,0.5, p2r(radius_length))
# Smooth the curve 
ker <- matrix(1,3,3)
# Calculate focal ("moving window") values for the neighborhood of focal cells using a matrix of weights, perhaps in combination with a function.
smooth_chm <- raster::focal(chm, w = ker, fun = mean, na.rm = TRUE)
# ws:Length or diameter of the moving window used to the detect the local maxima in the unit of the input data (usually meters)
# hmin:min height of a tree.
ttops <- tree_detection(smooth_chm, lmf(ws=3, hmin=2))
# A SpatialPointsDataFrame with an attribute Z for the tree tops and treeID with an individual ID for each tree.

x = plot(lasData)

add_treetops3d(x, ttops)
ttops



par(mfrow=c(1,2))
plot(chm)

plot(se,col=gray((max(se$Height_m)-se$Height_m)/(max(se$Height_m)-min(se$Height_m))),axes=T)

#crown area in square meters (CA_m2).
se@data

