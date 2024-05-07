# This file contains helper functions for describing arm waves from raw 
# coordinates and start/end tables.

#function to normalize vector; squares value to remove negatives and the take square root
norm_vec <- function(x) {
# calculates length of a vector
  sqrt(sum(x ^ 2))
} 

rad2deg <- function(rad) {(rad * 180) / (pi)} # conversion radians to degrees

deg2rad <- function(deg) {(deg * pi) / (180)} # conversion degrees to radians

split_endStart <- function(trial.data) {
  # Replaces "end/start" in trial.data with "end", duplicate row at row+1, 
  # then replaces "end" with "start". Controls for rows in which both left and 
  # right leg are "end/start"
  
  # first calculate number of rows in the final dataframe after duplicating 
  # end/start rows
  endrows <-
    nrow(trial.data) +
    length(grep("end/start", trial.data$left)) +
    length(grep("end/start", trial.data$right)) -
    sum(
      sapply(trial.data[2], function(x)
        trial.data[2] == "end/start" & trial.data[3] == "end/start")
    )
  
  # tease apart "end/start" rows
  for (i in 1:endrows) {
    if (trial.data[i, 2] == "end/start" &
        trial.data[i, 3] == "end/start") {
      trial.data[i, 2] <- "end"
      trial.data[i, 3] <- "end"
      trial.data <-
        rbind(trial.data[1:i, ], trial.data[i, ], trial.data[-(1:i), ])
      trial.data[i + 1, 2] <- "start"
      trial.data[i + 1, 3] <- "start"
    } else if (trial.data[i, 2] == "end/start" &
               trial.data[i, 3] != "end/start") {
      trial.data[i, 2] <- "end"
      trial.data <-
        rbind(trial.data[1:i, ], trial.data[i, ], trial.data[-(1:i), ])
      trial.data[i + 1, 2] <- "start"
    } else if (trial.data[i, 2] != "end/start" &
               trial.data[i, 3] == "end/start") {
      trial.data[i, 3] <- "end"
      trial.data <-
        rbind(trial.data[1:i, ], trial.data[i, ], trial.data[-(1:i), ])
      trial.data[i + 1, 3] <- "start"
    }
  }

  return(trial.data)
}

split.sides <- function(trial.data) {
  # split into left and right leg waves for convenience later
  lefts <- filter(trial.data, left != "0")
  lefts <- lefts[, c(1:2, 4:ncol(lefts))] # remove right column
  colnames(lefts)[2] <- "event"
  # removes duplicated rows introduced when other leg had an end/start frame:
  lefts <- unique(lefts) 
  
  rights <- filter(trial.data, right != "0")
  rights <- rights[, c(1, 3:ncol(rights))] # remove right column
  colnames(rights)[2] <- "event"
  rights <- unique(rights)
  
  waves.sides <- list(lefts, rights)
  return(waves.sides)
}

wave_angle.abs <- function(coxa.start, claw.start, coxa.end, claw.end) {
  # calculates change in "absolute" arm angle (angle between coxa-claw vector 
  # and horizontal plane) during a wave. 
  # Takes xyz coordinates of two point pairs.
  
    horiPlane <- c(0, 0, 1) # orthogonal vector to horizontal plane
    vector.start <- claw.start - coxa.start
    vector.end <- claw.end - coxa.end
    
    angle.start <-
      rad2deg(asin(
        norm_vec(horiPlane * vector.start) / norm_vec(vector.start) * 
          norm_vec(horiPlane)
      ))
    angle.end <-
      rad2deg(asin(
        norm_vec(horiPlane * vector.end) / norm_vec(vector.end) * 
          norm_vec(horiPlane)
      ))
    
    amplitude <- angle.end - angle.start
    return(amplitude)
  }

# functions to calculate visual angle moved from female perspective ------------
xyz.matrix <- function(trial.data.row){
  # 
  X <- as.numeric(trial.data.row[, seq(4, length(trial.data.row)-1, 3)])
  Y <- as.numeric(trial.data.row[, seq(5, length(trial.data.row)-1, 3)])
  Z <- as.numeric(trial.data.row[, seq(6, length(trial.data.row)-1, 3)])
  
  points <- matrix(c(X,Y,Z), nrow = 12, ncol = 3)
  colnames(points) <- c("X", "Y", "Z")
  return(points)
}   #1: Makes a 12x3 matrix from one row of trial.data

male_azimuth <- function(points){
  # This function calculates male "camera" angle vs world x-axis. 
  # Takes data from one frame as a [12,3] xyz array
  
  male <- c(points[1,1], points[1,2], 1)
  vertPlane <- c(0, 1, 0) # orthogonal vector to horizontal plane
  
  
  # calculates "camera" angle vs world x-axis.
  camera <-
    asin(norm_vec(vertPlane * male) / norm_vec(male) * norm_vec(vertPlane))
  
  # determines quadrant of vector
  if (male[1] > 0) {
    if (male[2] > 0) {
      q <- 1
    } else {
      q <- 4
    }
  } else if (male[2] > 0) {
    q <- 2
  } else{
    q <- 3
  }
  
  # corrects the angle depending on quadrant
  if (q == 1) {
    camera <- (2 * pi) - camera
  } else if (q == 2) {
    camera <- pi + camera
  } else if (q == 3) {
    camera <- pi - camera
  }
  
  return(camera)
}    
#2: Calculates male "camera" angle vs world x-axis. 
#   Takes data from one frame as a [12,3] xyz array

zrotate <- function(point, angle) {
  # this function rotates an xyz point (or array of xyz points) counter-
  # clockwise around the z-axis. 
  rotator <- matrix(
    data = c(cos(angle), -sin(angle), 0,
             sin(angle), cos(angle), 0,
             0, 0, 1),
    nrow = 3,
    ncol = 3
  ) 
  
  point_rot <- point %*% rotator
  return(point_rot)
}    
#3: rotates an xyz point (or array of xyz points) counter-clockwise around the z-axis.

create_rotated_view <-  function(trial.data){
  # set up empty array to hold rotated rows:
rotated.frames <- matrix(nrow = nrow(trial.data), ncol = 36) 


  for(i in 1:nrow(trial.data)){
    row.as.matrix <- xyz.matrix(trial.data[i,])
    rotation_angle <- male_azimuth(row.as.matrix)
    rotated.points <- zrotate(row.as.matrix, rotation_angle)
    #turns 12x3 matrix of rotated points back into a row vector:
    rotated.frames[i,] <-  as.vector(t(rotated.points)) 
  }

colnames(rotated.frames)<- paste(rep("pt", 36), rep(1:12, each = 3),
                                 rep(c("_rX", "_rY", "_rZ"), 12),sep="")

trial.data <- cbind(trial.data, rotated.frames)
trial.data <- trial.data[,c(1,2,3,40,4:39,41:76)] # gets the fps column to front
return(trial.data)
} 
#4: takes all rows from trial.data and adds female view coordinates.

visual_angle <- function(coxa.start.r, claw.start.r, coxa.end.r, claw.end.r){
  # script takes xyz coords of each point (3 length vector). Then does a 
  # projection where, in this rotated orientation, e.g. 
  # pt3_rX AKA coxa.start.r[1] becomes the depth coordinate for scaling, 
  # pt3_rY becomes x, and pt3_rZ becomes y.
  
  # with female position as (0,0,0):
  coxa.start.proj_x <- coxa.start.r[2] / coxa.start.r[1]
  coxa.start.proj_y <- coxa.start.r[3] / coxa.start.r[1]
  coxa.end.proj_x <- coxa.end.r[2] / coxa.end.r[1]
  coxa.end.proj_y <- coxa.end.r[3] / coxa.end.r[1]
  
  claw.start.proj_x <- claw.start.r[2] / claw.start.r[1]
  claw.start.proj_y <- claw.start.r[3] / claw.start.r[1]
  claw.end.proj_x <- claw.end.r[2] / claw.end.r[1]
  claw.end.proj_y <- claw.end.r[3] / claw.end.r[1]
  
  claw.start.proj <- c(claw.start.proj_x, claw.start.proj_y)
  claw.end.proj <- c(claw.end.proj_x, claw.end.proj_y)
  
  clawmove <- claw.end.proj - claw.start.proj # 2D vector of claw movement
  
  # the "canvas" is 1 unit (mm) away from the eye, thus angle moved is just 
  # inverse tangent of distance moved:
  angle_moved <- rad2deg(atan(norm_vec(clawmove)))
  
return(angle_moved)
} #5: calculates visual angle moved by left and right leg

# reshape left/right start/end data into a frame that has one wave per row -----

build.waves <- function(lefts, rights) {
  n.waves.left <- nrow(lefts) / 2
  n.waves.right <- nrow(rights) / 2
  n.waves <- n.waves.left + n.waves.right
  
  
  # inititate data frame, one row per wave
  waves <- data.frame(
    leg = character(n.waves),
    start = integer(n.waves),
    end = integer(n.waves),
    duration = numeric(n.waves),
    amplitude.male = numeric(n.waves),
    velocity.male = numeric(n.waves),
    visual.angle = numeric(n.waves),
    visual.velocity = numeric(n.waves),
    stringsAsFactors = FALSE
  )
  
  # populate "waves" dataframe with data from left leg waves, calculate 
  # amplitude and velocity in relation to male
  i <- 1
  j <- 1
  while (i <= n.waves.left) {
    while (j <= nrow(lefts) - 1) {
      waves$leg[i] <- "left"
      waves$start[i] <- lefts$frame[j]
      waves$end[i] <- lefts$frame[j + 1]
      
      # coordinates of relevant joints
      coxa.start <-
        c(lefts$pt3_X[j], lefts$pt3_Y[j], lefts$pt3_Z[j])
      coxa.end <-
        c(lefts$pt3_X[j + 1], lefts$pt3_Y[j + 1], lefts$pt3_Z[j + 1])
      claw.start <-
        c(lefts$pt6_X[j], lefts$pt6_Y[j], lefts$pt6_Z[j])
      claw.end <-
        c(lefts$pt6_X[j + 1], lefts$pt6_Y[j + 1], lefts$pt6_Z[j + 1])
      
      distance <- 
        sqrt(((lefts$pt11_X[j] - lefts$pt1_X[j])^2) + 
             ((lefts$pt11_Y[j] - lefts$pt1_Y[j])^2))
      
      duration <- (lefts$frame[j + 1] - lefts$frame[j]) / lefts$fps[j] # seconds
      amplitude <- wave_angle.abs(coxa.start, claw.start, coxa.end, claw.end)
      velocity <- sqrt((amplitude/duration)^2)
      
      # joint coordinates in rotated reference frame (for 2D projection)
      coxa.start.r <-
        c(lefts$pt3_rX[j], lefts$pt3_rY[j], lefts$pt3_rZ[j])
      coxa.end.r <-
        c(lefts$pt3_rX[j + 1], lefts$pt3_rY[j + 1], lefts$pt3_rZ[j + 1])
      claw.start.r <-
        c(lefts$pt6_rX[j], lefts$pt6_rY[j], lefts$pt6_rZ[j])
      claw.end.r <-
        c(lefts$pt6_rX[j + 1], lefts$pt6_rY[j + 1], lefts$pt6_rZ[j + 1])
      eye.start <- lefts$pt11_rX[j]
      eye.end <- lefts$pt11_rX[j+1]
      
      angle.female <- visual_angle(coxa.start.r,
                                   claw.start.r,
                                   coxa.end.r,
                                   claw.end.r)
      
      velocity.female <- sqrt((angle.female/duration)^2)
      
      waves$distance[i] <- distance
      waves$duration[i] <- duration
      waves$amplitude.male[i] <- amplitude
      waves$velocity.male[i] <- velocity
      waves$visual.angle[i] <- angle.female
      waves$visual.velocity[i] <- velocity.female
      
      i <- i + 1
      j <- j + 2
    }
  }
  
  i <- n.waves.left + 1
  j <- 1
  while (i <= n.waves) {
    while (j <= nrow(rights) - 1) {
      waves$leg[i] <- "right"
      waves$start[i] <- rights$frame[j]
      waves$end[i] <- rights$frame[j + 1]
      
      # coordinates of relevant joints
      coxa.start <-
        c(rights$pt7_X[j], rights$pt7_Y[j], rights$pt7_Z[j])
      coxa.end <-
        c(rights$pt7_X[j + 1], rights$pt7_Y[j + 1], rights$pt7_Z[j + 1])
      claw.start <-
        c(rights$pt10_X[j], rights$pt10_Y[j], rights$pt10_Z[j])
      claw.end <-
        c(rights$pt10_X[j + 1], rights$pt10_Y[j + 1], rights$pt10_Z[j + 1])
      
      distance <- sqrt(((rights$pt11_X[j] - rights$pt1_X[j])^2) + 
                         ((rights$pt11_Y[j] - rights$pt1_Y[j])^2))
      
      duration <- (rights$frame[j + 1] - rights$frame[j]) / rights$fps[j]
      amplitude <- wave_angle.abs(coxa.start, claw.start, coxa.end, claw.end)
      velocity <- sqrt((amplitude/duration)^2)
      
      # joint coordinates in rotated reference frame (for 2D projection)
      coxa.start.r <-
        c(rights$pt7_rX[j], rights$pt7_rY[j], rights$pt7_rZ[j])
      coxa.end.r <-
        c(rights$pt7_rX[j + 1], rights$pt7_rY[j + 1], rights$pt7_rZ[j + 1])
      claw.start.r <-
        c(rights$pt10_rX[j], rights$pt10_rY[j], rights$pt10_rZ[j])
      claw.end.r <-
        c(rights$pt10_rX[j + 1], rights$pt10_rY[j + 1], rights$pt10_rZ[j + 1])
      eye.start <- rights$pt11_rX[j]
      eye.end <- rights$pt11_rX[j+1]
      
      angle.female <- visual_angle(coxa.start.r,
                                   claw.start.r,
                                   coxa.end.r,
                                   claw.end.r)
      
      velocity.female <- sqrt((angle.female/duration)^2)
      
      waves$distance[i] <- distance
      waves$duration[i] <- duration
      waves$amplitude.male[i] <- amplitude
      waves$velocity.male[i] <- velocity 
      waves$visual.angle[i] <- angle.female
      waves$visual.velocity[i] <- velocity.female
      
      i <- i + 1
      j <- j + 2
    }
  }
  
  # add ID, condition, and wave ID columns
  waves$id <- str_sub(trial.curr, 1, 4)
  waves$treat <- str_sub(trial.curr, 6, -1)
  return(waves)
}

plot.arena <- function(points) {
  # rotatable 3D plot of one frame. Requires 12 points as 12x3 matrix as 
  # generated by the "xyz.matrix" function above.
  
  # set up arena circle ----
  n <- 300 
  theta <- seq(0, 2*pi, len=n) 
  circle_x <- cos(theta) 
  circle_y <- sin(theta) 
  circle_z <- rep(0, n) 
  
  open3d() # open rgl device
  plot3d(
    points,
    radius = 1,
    col = rainbow(12),
    xlim = c(-50, 50),
    ylim = c(-50, 50),
    zlim = c(0, 100)
  ) # draw 12 points
  
  # draw 50mm circle to represent arena:
  lines3d(50 * circle_x, 50 * circle_y, circle_z) 
}

