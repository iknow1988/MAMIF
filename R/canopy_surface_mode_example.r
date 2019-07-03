library(lidR)
#https://www.rdocumentation.org/packages/lidR/versions/2.0.2/topics/grid_canopy

LASfile <- system.file("extdata", "MixedConifer.laz", package="lidR")
las <- readLAS(LASfile)
col <- height.colors(50)
las
plot(las)

# Points-to-raster algorithm with a resolution of 1 meter
chm <- grid_canopy(las, res = 1, p2r())
print(dim(chm))
plot(chm, col = col)


chm <- grid_canopy(las, res = 0.5, p2r())
print(dim(chm))
plot(chm, col = col)

# Points-to-raster algorithm with a resolution of 0.5 meters replacing each
# point by a 20-cm radius circle of 8 points
chm <- grid_canopy(las, res = 0.5, p2r(0.4))
plot(chm, col = col)

# Basic triangulation and rasterization of first returns
chm <- grid_canopy(las, res = 0.5, dsmtin())
plot(chm, col = col)

# Khosravipour et al. pitfree algorithm
chm <- grid_canopy(las, res = 0.5, pitfree(c(0,2,5,10,15), c(0, 1.5)))
plot(chm, col = col)


las <- readLAS(LASfile, select = "xyz", filter = "-drop_z_below 0")
col <- pastel.colors(200)

chm <- grid_canopy(las, res = 0.5, p2r(0.3))
ker <- matrix(1,3,3)
chm <- raster::focal(chm, w = ker, fun = mean, na.rm = TRUE)

ttops <- tree_detection(chm, lmf(4, 2))
las   <- lastrees(las, silva2016(chm, ttops))
plot(las, color = "treeID", colorPalette = col)
ttops
ttops = tree_detection(chm, lmf(3))
crowns = silva2016(chm, ttops)()
plot(crowns )
