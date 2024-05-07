# this is a testing ground for 2-D projection of raw absolute xyz coordinates. 

library(rgl)
library(readr)

# read first start and end of exemplar xyzpts file
exemplar <- read_csv("~/Documents/Work/Postdoc Pitt/Projects/2016 NSF projects/H4 Long range displays vs Environment/Arena analysis/exemplars/2625_Daniel_xyzpts.csv")
# exemplar <- exemplar[2,]
exemplar <- trial.data[1, 4:39]

# coordinates of 12 points as numeric vectors
X <- as.numeric(exemplar[seq(1, length(exemplar), 3)])
Y <- as.numeric(exemplar[seq(2, length(exemplar), 3)])
Z <- as.numeric(exemplar[seq(3, length(exemplar), 3)])

points <- matrix(c(X,Y,Z), nrow = 12, ncol = 3)
colnames(points) <- c("X", "Y", "Z")

## plotting ----

# plot 12 points before any rotations -----
plot.arena <- function(points) {
  
  # set up arena circle ----
  n <- 300 
  theta <- seq(0, 2*pi, len=n) 
  circle_x <- cos(theta) 
  circle_y <- sin(theta) 
  circle_z <- rep(0, n) 
  
  open3d() # open rgl device
  plot3d(
    points,
    col = rainbow(12),
    xlim = c(-50, 50),
    ylim = c(-50, 50),
    zlim = c(0, 100)
  ) # draw 12 points
  lines3d(50 * circle_x, 50 * circle_y, circle_z) # draw 50mm circle to represent arena
}

# female orientation angle vs x-axis

female_azimuth <- function(points) {
  # this function calculates female "camera" angle vs world x-axis. Takes data from one frame as a [12,3] xyz array.
  # However, this doesn't respect viewing direction, need to add quadrants.
  
  pt.11 <- c(points[11, 1], points[11, 2], 1)
  pt.12 <- c(points[12, 1], points[12, 2], 1)
  vertPlane <- c(0, 1, 0) # orthogonal vector to horizontal plane
  fem <- pt.11 - pt.12
  
  # calculates "camera" angle vs world x-axis.
  camera <-
    asin(norm_vec(vertPlane * fem) / norm_vec(fem) * norm_vec(vertPlane))
  
  # determines quadrant of vector
  if (fem[1] > 0) {
    if (fem[2] > 0) {
      q <- 1
    } else {
      q <- 4
    }
  } else if (fem[2] > 0) {
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

# rad2deg(camera)

######
zrotate <- function(point, angle) {
# this function rotates an xyz point (or array of xyz points) counter-clockwise around the z-axis. This brings 11 and 12 in line with the x-axis.
  
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

camera <- female_azimuth(points)
points_rotated <-
  zrotate(points, camera) # rotates points by the "camera" angle so that female viewing direction coincides with world x-axis

# plot rotated points
plot.arena(points_rotated)

# now columns 2 and 3 of points_rotated correspond to x and y coordinates in camera space. Column 1 is the distance from female

# 2D projection:
proj_x <-
  points_rotated[1:10, 2] / (points_rotated[1:10, 1] - points_rotated[11, 1]) # projected on canvas 1mm female eye (point 11)
proj_y <-
  points_rotated[1:10, 3] / (points_rotated[1:10, 1] - points_rotated[11, 1])

plot(proj_x[1:10], proj_y[1:10], xlim = c(-2, 2), ylim = c(0, 2))

# now check axis directions, if it works around the circle, and run some test numbers on waves

