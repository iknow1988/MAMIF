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
setwd('/Users/ethanchen/Desktop/2019SEM1/DataProject/MAMIF/R/Shiny_app_version1')
library(shiny)
library(lidR)
library(dplyr)
library(rgl)

source('../utils.R')

#### load and only use xyz coordinate
dat = lidR::readLAS(files = "../../Shared_repo/SP03_Stony.las" ,select = "xyz")

dataPath <- "/Users/ethanchen/Desktop/2019SEM1/DataProject/Shared_repo/"

#small_dat <- pre_processing(las_df = dat ,boundary = 15  , shrunk_factor = 0.2)

#small_dat_with_ground <- lasground(small_dat, csf())

#small_dat_with_ground@data
# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Section 1 : data preprocessing"),
    
    
    h3("Loading las file."),
   # Input: Selector for choosing dataset ----
   selectInput(inputId = "dataset",
               label = "Choose a dataset:",
               choices = c("SP09_Coalmine_ck.las", "SP14_Garvey.las", "SP03_Stony.las")),
   # Sidebar panel for inputs ----
   sidebarPanel(
       
       # Input: Slider for the number of bins ----
       sliderInput(inputId = "distanceFromCenter",
                   label = "Distance from center (meter):",
                   min = 1,
                   max = 50,
                   value = 10)
       
   ),

       
   mainPanel(
       
       rglwidgetOutput("plot",  width = 800, height = 600)    
   )
              
    
)

# Define server logic required to draw a histogram
server <- function(input, output) {


    output$plot <- renderRglwidget({
        
        distance = input$distanceFromCenter
        rgl.open(useNULL=T)
        
        dat = lidR::readLAS(files = paste(dataPath,input$dataset  , sep='') ,select = "xyz")
        small_dat <- pre_processing(las_df = dat ,boundary = distance  , shrunk_factor = 0.02)
        
        
        small_dat <- lasground(small_dat, csf())
        
        points3d(x = small_dat@data$X , y = small_dat@data$Y , z = small_dat@data$Z , color=small_dat@data$Classification)
        
        axes3d()
        rglwidget() 
        
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
