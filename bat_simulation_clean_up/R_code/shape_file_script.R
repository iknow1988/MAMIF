
library(lidR)
library(dplyr)
library(TreeLS)
library(maptools)
source("./utils.R")

install.packages('lidR')
install.packages('dplyr')
install.packages('TreeLS')
install.packages('maptools')
###########################
# Data preprocessing
###########################
SITE_WIDTH = 40
SHRUNK_FACTOR = 0.15
HEIGHT_THRESHOOD = 10
setwd('/Users/ethanchen/Desktop/2019SEM1/MAMIF/bat_simulation_clean_up/R_code')
dataPath <- "/Users/ethanchen/Desktop/2019SEM1/LAS"

shapeFileCreation <- function(file_name){
   
   file_path <- paste(dataPath ,file_name , sep = '')
   
   dat <- lidR::readLAS(file_path)
   
   small_dat <- pre_processing(las_df = dat, boundary =  SITE_WIDTH/2, shrunk_factor = SHRUNK_FACTOR)
   
   small_dat_with_ground <- lasground(small_dat, csf())
   
   
   dtm <- grid_terrain(small_dat_with_ground ,algorithm = knnidw(k = 6L, p = 2))
   small_dat_with_ground <- small_dat_with_ground - dtm
   small_dat_no_ground <- filter_class(small_dat_with_ground , class_index = 1)
   
   
   tls = small_dat_no_ground@data %>% filter(Z < HEIGHT_THRESHOOD)
   tls = LAS(tls)
   # normalize the point cloud
   #tls = tlsNormalize(tls, keepGround = T)
   #plot(tls, color='Classification')
   
   # extract the tree map from a thinned point cloud
   thin = tlsSample(tls, voxelize(0.05))
   map = treeMap(thin, map.hough(min_density = 0.001,max_radius = 0.3))
   
   ############## Multiple
   ids<-unique(map@data$TreeID)
   polygon_list = as.list(rep(NA,length(ids)))
   
   
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
   
   
   
   setwd('/Users/ethanchen/Desktop/2019SEM1/MAMIF/bat_simulation_clean_up/R_code/shape_files')
   output_name = gsub('.las','',file_name)
   writeSpatialShape(spatial_df , output_name)
}


#shapeFileCreation('SP09_Coalmine_ck.las')

for(f in list.files(dataPath)){
   print(f)
   shapeFileCreation(f)
}
