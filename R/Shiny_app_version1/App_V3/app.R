#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#


#-------------------
# https://github.com/Jean-Romain/PointCloudViewer
#
setwd("/Users/ethanchen/Desktop/2019SEM1/DataProject/MAMIF/R/Shiny_app_version1")
library(shiny)
library(lidR)
library(dplyr)
library(rgl)
# To plot raster layer in 3D.
library(rasterVis)

library("styler")
source("../utils.R")

#### load and only use xyz coordinate
dat <- lidR::readLAS(files = "../../Shared_repo/SP03_Stony.las", select = "xyz")

dataPath <- "/Users/ethanchen/Desktop/2019SEM1/DataProject/Shared_repo/"



# 06-navlist.R

library(shiny)

ui <- navbarPage(title = "Phase 1 Forest Data Analysis",
                 
                ######################################################
                ## START SECTION 1 Preprocessing
                ######################################################
                 tabPanel(
                     # Application title
                     title = "Section 1 : data preprocessing",
                     
                     # Sidebar panel for inputs ----
                     sidebarPanel(width=6,
                                  h3("Loading las file."),
                                  # Input: Selector for choosing dataset ----
                                  selectInput(
                                      inputId = "dataset",
                                      label = "Choose a dataset:",
                                      choices = c("SP09_Coalmine_ck.las", "SP14_Garvey.las", "SP03_Stony.las")
                                  ),
                                  
                                  # Input: Slider for the number of bins ----
                                  sliderInput(
                                      inputId = "distanceFromCenter",
                                      label = "Distance from center (meter):",
                                      min = 1,
                                      max = 50,
                                      value = 10
                                  )
                     ),
                     
                     mainPanel(
                         h2("Filter the ground"),
                         rglwidgetOutput("filtered_with_ground" , width = "600px", height = "600px"),    
                         h1("Flattern the terrian"),
                         h2("DTM"),
                         plotOutput(outputId = 'dtm'),
                         h2("Flattern"),
                         rglwidgetOutput("flatterned")
                         
                     )
                    
                 ),
                 
                
                ######################################################
                ## START SECTION 2 Canopy height model 
                ######################################################
                 
                 tabPanel(title = "Section 2 : canopy height model",
                          
                          p('Points-to-raster algorithm with a resolution of "q" meters replacing each
# point by a "r" meter radius circle of 8 points , higher the radius_length ,smoother the surface of the raster layer.'),
                          tabPanel(
                              # Application title
                              title = "Section 2 : Canopy height model",
                              
                              # Sidebar panel for inputs ----
                              sidebarPanel(width=6,
                                          
                                           # Input: Slider for the number of bins ----
                                           sliderInput(
                                               inputId = "chm_radius_length",
                                               label = "Radius Length ( r ) in meters",
                                               min = 0.1,
                                               max = 5,
                                               value = 0.2
                                           ), # Input: Slider for the number of bins ----
                                           sliderInput(
                                               inputId = "chm_resolution",
                                               label = "Resolution ( q ) in meters",
                                               min = 0.1,
                                               max = 1,
                                               value = 0.5
                                           )
                              )
                        ),
                        mainPanel(
                            h2("Canopy height model 2D"),
                            plotOutput(outputId = "canopy_height_model"),
                            h2("Canopy height model 3D"),
                            rglwidgetOutput(outputId = "canopy_height_model_3D" , height = "700px",width = "700px"),
                            h2("Tree counting (First attempt)"),
                            p("It implements an algorithm for tree detection based on a local maximum filter
Problem : changing the parameter ws (size of the moving window used to detect the local maxima) effect the number of treeID in the output. The higher the ws the lesser number of tree it identifys. However, the lower the ws, it identifies a tree a multiple trees."),
                            p('ws:Length or diameter of the moving window used to the detect the local maxima in the unit of the input data (usually meters)'),
                            
                            
                            
                            sidebarPanel(width=12,
                                         
                                         # Input: Slider for the number of bins ----
                                         sliderInput(
                                             inputId = "ws",
                                             label = "Size of moving window ( WS ) in meters",
                                             min = 1,
                                             max = 10,
                                             value = 4
                                         )
                            ),
                            h2(textOutput("number_of_trees"))
                            ,
                            rglwidgetOutput(outputId = "count_tree_v1" , height = "700px",width = "700px")
                    
                            
                        )
                 )
                 
                
)
                 


server <- function(input, output) {
    
    ######################################################
    ## START SECTION 1 Preprocessing
    ######################################################
    
    lasData <- reactive({
        lidR::readLAS(files = paste(dataPath, input$dataset, sep = ""))
    })
    
    smallData <- reactive({
        tmp <- pre_processing(las_df = lasData(), boundary = input$distanceFromCenter, shrunk_factor = 0.02)
        print("Input")
        print(input)
        
        tmp <- lasground(tmp, csf())
        return(tmp)
    })
    
    dtm <- reactive({
        grid_terrain(smallData() ,algorithm = knnidw(k = 6L, p = 2))
    })
    
    flatten_data <- reactive({
        d <- dtm()
        data <- smallData()
        data - d
    })
    
    
    output$filtered_with_ground <- renderRglwidget({
        rgl.open(useNULL = T)

        small_dat <- smallData()
        
        points3d(x = small_dat@data$X, y = small_dat@data$Y, z = small_dat@data$Z, color = small_dat@data$Classification)
        
        axes3d()
        rglwidget()
    })
    
    output$dtm <- renderPlot({
        
        plot(dtm())
    })
    
    output$flatterned <- renderRglwidget({
        rgl.open(useNULL = T)
        dat <- flatten_data()
        points3d(x = dat@data$X, y = dat@data$Y, z = dat@data$Z)
        axes3d()
        rglwidget()
        
    })
    
    ######################################################
    ## START SECTION 2 CHM
    ######################################################
    
    chm <- reactive({
        grid_canopy(flatten_data() ,input$chm_resolution, p2r(input$chm_radius_length))
    })
    
    output$canopy_height_model <- renderPlot({
        dat <- chm()
        col <- height.colors(30)
        plot(dat , col=col)
    })
    
    output$canopy_height_model_3D <- renderRglwidget({
        rgl.open(useNULL = T)
        dat <- chm()
       
        plot3D(dat)
        rglwidget()
    })
    
    output$count_tree_v1 <- renderRglwidget({
        rgl.open(useNULL = T)
        dat <- flatten_data()
        
        ttops <- tree_detection(dat, lmf(ws=input$ws, hmin=2))
        # A SpatialPointsDataFrame with an attribute Z for the tree tops and treeID with an individual ID for each tree.
        
        x<-plot(dat)
        
        add_treetops3d(x, ttops)
        
        output$number_of_trees <- renderText({
            paste("Number of trees : " , length(ttops$treeID) )
        })
        
        rglwidget()
    })
    
    
    
    
 
}

shinyApp(server = server, ui = ui)

