
library(lidR)
library(dplyr)
library(TreeLS)
library(maptools)
source("utils.R")
setwd('/Users/ethanchen/Desktop/2019SEM1/MAMIF/bat_simulation_clean_up/R_code')
getwd()

###########################
# Data preprocessing
###########################
SITE_WIDTH = 10
SHRUNK_FACTOR = 0.15
LOW_THRESHOOD = 5
HIGH_THRESHOOD = 8

dataPath <- "/Users/ethanchen/Desktop/2019SEM1/MAMIF/bat_simulation_clean_up/R_code/LAS/"
output_dir = 'shape_files'
shapeFileCreation <- function(file_name){
   
   file_path <- paste(dataPath ,file_name , sep = '')
   print(file_path)
   dat <- lidR::readLAS(file_path)
   
   small_dat <- pre_processing(las_df = dat, boundary =  SITE_WIDTH/2, shrunk_factor = SHRUNK_FACTOR)
   
   small_dat_with_ground <- lasground(small_dat, csf())
   
   
   dtm <- grid_terrain(small_dat_with_ground ,algorithm = knnidw(k = 6L, p = 2))
   small_dat_with_ground <- small_dat_with_ground - dtm
   small_dat_no_ground <- filter_class(small_dat_with_ground , class_index = 1)
   plot(small_dat_no_ground)
   
  # tls = small_dat_no_ground@data %>% filter(Z < HIGH_THRESHOOD & Z > LOW_THRESHOOD)
   #tls = LAS(tls)
   # normalize the point cloud
   #tls = tlsNormalize(tls, keepGround = T)
   #plot(tls, color='Classification')
   tls = small_dat_no_ground
   # extract the tree map from a thinned point cloud
   thin = tlsSample(tls, voxelize(0.05))
   map = treeMap(thin, map.hough(min_density = 0.001,max_radius = 0.3,hmin = LOW_THRESHOOD,hmax = HIGH_THRESHOOD))
   plot(map)
   
   ############## Multiple
   ids<-unique(map@data$TreeID)
   polygon_list = as.list(rep(NA,length(ids)))
   print(paste("Paste number of trees ",length(ids)))
   
   for (i in seq_along(ids)){
      id <- ids[i]
      sub_df <- map@data %>% filter(TreeID == id) %>% select(c(X,Y))
      index <- chull(sub_df)
      po<- Polygon(sub_df[index,])
      po1 <- Polygons(list(po) , as.character(i))
      polygon_list[[i]] <- po1
   }
   
   
   sp <- SpatialPolygons(polygon_list)
   
   df <- data.frame(ID=character(), stringsAsFactors=FALSE )
   for (i in sp@polygons ) { 
      df <- rbind(df, data.frame(ID=i@ID, stringsAsFactors=FALSE)) 
   }
   
   row.names(df) <- df$ID
   spatial_df <- SpatialPolygonsDataFrame(sp, df)
   
   
   plot(spatial_df)
   #setwd('/Users/ethanchen/Desktop/2019SEM1/MAMIF/bat_simulation_clean_up/R_code/shape_files')
   output_name = gsub('.las','',file_name)
   #writeSpatialShape(spatial_df,paste(output_dir,output_name,sep='/') )
   writeSpatialShape(spatial_df,output_name)
   return(length(ids))
}


N = shapeFileCreation('84ish seaview.las')
 
N
names = c()
n_trees = c()

arr = c('84ish seaview.las',   '85 seaview.las' ,  '94 seaview.las' ,  '95 seaview.las',   'SP02 Big Hill.las')


for(f in arr){
   print(f)
   names <- c(names , f)
   n_tree <- shapeFileCreation(f)
   
   n_trees <- c(n_trees , n_tree)
}


for(f in list.files(dataPath)){
   print(f)
   names <- c(names , f)
   n_tree <- shapeFileCreation(f)
   
   n_trees <- c(n_trees , n_tree)
}
getwd()

D <- data.frame(names , n_trees)

D

write.csv(D,'N_trees_for_each_site_width60_5_to_8.csv')






