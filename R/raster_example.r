library(raster)
x <- raster()
x
x <- raster(ncol=36, nrow=18, xmn=-1000, xmx=1000, ymn=-100, ymx=900)
x
res(x)

# change the resolution
res(x) <- 100
res(x)

ncol(x)
# change the numer of columns (affects resolution)
ncol(x) <- 18
res(x)
# set the coordinate reference system (CRS) (define the projection)

projection(x) <- "+proj=utm +zone=48 +datum=WGS84"
x
r <- raster(ncol=10, nrow=10)
ncell(r)
# add value to raster
values(r) <- 1:ncell(r)
hasValues(r)

values(r)[1:10]
plot(r, main='Raster with 100 cells')
res(r)

LASfile <- system.file("extdata", "MixedConifer.laz", package="lidR")
las = readLAS(LASfile)
# Full manual tree finding
ttops = tree_detection(las, manual())
