# rename arena video files according to list

library(stringr)

file_location <- "/Volumes/number1hfs/GoPro Arena/Experiments/oregonensis"
setwd(file_location)
male_list <- read.csv("Arena trials oregonensis 2017-10-12.csv")[,1]
camera <- "D"
file_list <- dir(camera)

for (i in 1:length(file_list)) {
  file.rename(paste0(camera, "/", file_list[i]),
              paste0(camera, "/",
                camera,
                "_",
                male_list[i], "_",
                file_list[i])
              )
}
