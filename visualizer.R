trial = "1866 1.5x"
side = "left"
start_frame = 2785
end_frame = 2823

visualize_frame <- function(trial, side, start_frame, end_frame){
# reads a trial and plots 3D points as well as a 2D projection
# create file paths ----
  file.points <- list.files(paste0("Data/", trial), pattern = "_xyzpts.csv")     # find xyz point file
  file.StaEnd <- list.files(paste0("Data/", trial), pattern = "_StartEnd.xlsx")  # find start/end frames file
  file.CI <- list.files(paste0("Data/", trial), pattern = "_xyzCI.csv")          # find xyz point file
  
  file.points <- paste0("Data/", trial,"/",file.points)                          # assemble file path
  file.StaEnd <- paste0("Data/", trial,"/",file.StaEnd)
  file.CI <- paste0("Data/", trial,"/",file.CI)
  
  # load data files and create frames holding coordinates of left and right waves ====
  trial.data <-
    read.csv(
      file.points,
      header = TRUE,
      stringsAsFactors = FALSE
    )
  
  trial.CI <-
    read.csv(
      file.CI,
      header = TRUE,
      stringsAsFactors = FALSE
    )
  
  start.end.frames <- read_excel(path = file.StaEnd)
  start.end.frames[is.na(start.end.frames)] <- 0 # change NAs to zeros
  
  # clean up DLT xyz and CI files by adding frame numbers and removing empty rows
  trial.data$frame<- 1:nrow(trial.data)
  trial.data<- na.omit(trial.data)
  
  trial.CI$frame<- 1:nrow(trial.CI)
  trial.CI<- na.omit(trial.CI)
  
  # clean up start/end file by omitting irrelevant columns and renaming the kept ones
  start.end.frames <- start.end.frames[, 2:4]
  colnames(start.end.frames) <- c("frame", "left", "right")
  
  # join xyz table with start/end frames
  trial.data <- inner_join(start.end.frames, trial.data, by = "frame")
  
  # add frame rate column -----
  if(trial %in% trials24) {
    trial.data$fps <- 24
  } else if (trial %in% trials30) {
    trial.data$fps <- 30
  } else {
    trial.data$fps <- 120
  }
  
  # add rotated points to trial.data -----
  trial.data <- create_rotated_view(trial.data)
  
  # deal with "end/start" values by duplicating those rows ----
  trial.data <- split_endStart(trial.data)
  
  # split waves into separate data frames for left and right for ease of use ----
  waves.sides <- split.sides(trial.data)
  lefts <- data.frame(waves.sides[1])
  rights <- data.frame(waves.sides[2])
  
  # calculate frame with one row per wave----------------------------------------------
  waves <- build.waves(lefts, rights)

  # 3D plot of start frame -----
  if (side == "left") {
    exemplar <-
      filter(trial.data, frame == start_frame & left == "start")
  } else {
    exemplar <-
      filter(trial.data, frame == start_frame & right == "start")
  }
  
  exemplar <- exemplar[1, 41:76]
  
  # coordinates of 12 points as numeric 12x3 matrix:
  X <- as.numeric(exemplar[, seq(1, length(exemplar), 3)])
  Y <- as.numeric(exemplar[, seq(2, length(exemplar), 3)])
  Z <- as.numeric(exemplar[, seq(3, length(exemplar), 3)])
  points <- matrix(c(X,Y,Z), nrow = 12, ncol = 3)
  colnames(points) <- c("X", "Y", "Z")

  plot.arena(points) # 3D plot 
  
  # 2D projection:
  proj_x <-
    -(points[1:10, 2] / (points[1:10, 1])) # projected on canvas 1mm female eye (point 11)
  proj_y <-
    points[1:10, 3] / (points[1:10, 1])
  
  plot(proj_x[1:10], proj_y[1:10], xlim = c(-1, 1), ylim = c(0, 2), pch = 16, cex = 0.5)
  
  # 3D plot of end frame -----
  if (side == "left") {
    exemplar <-
      filter(trial.data, frame == end_frame & left == "end")
  } else {
    exemplar <-
      filter(trial.data, frame == end_frame & right == "end")
  }
  
  exemplar <- exemplar[1, 41:76]
  
  # coordinates of 12 points as numeric 12x3 matrix:
  X <- as.numeric(exemplar[, seq(1, length(exemplar), 3)])
  Y <- as.numeric(exemplar[, seq(2, length(exemplar), 3)])
  Z <- as.numeric(exemplar[, seq(3, length(exemplar), 3)])
  points <- matrix(c(X,Y,Z), nrow = 12, ncol = 3)
  colnames(points) <- c("X", "Y", "Z")
  
  plot.arena(points) # 3D plot 
  
  # 2D projection:
  proj_x <-
    -(points[1:10, 2] / (points[1:10, 1])) # projected on canvas 1mm female eye (point 11)
  proj_y <-
    points[1:10, 3] / (points[1:10, 1])
  
  plot(proj_x[1:10], proj_y[1:10], xlim = c(-1, 1), ylim = c(0, 2), pch = 16, cex = 0.5)
  
}
