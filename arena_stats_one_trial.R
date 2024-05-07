# Plot single exemplar wave dataset. For exemplar enter prefix of DLTdv5 files, e.g. "2625_1.5x" for 2625_1.5x_xyzpts.csv

plot_one_trial <- function(exemplar){
  
library(dplyr)
library(tidyr)
library(ggplot2)
library(utils)
library(readxl)
library(stringr)

source("R/build_spider_fun.R")


#------------------------READ & WRANGLE DATA------------------------------------------------------#

build_spider_one_trial(exemplar)

waves <- read.csv(
  paste0("exemplars/", exemplar, "_waves.csv"),
  header = TRUE
)

# add down/upstroke column
waves$stroke <- ifelse(waves$amplitude.male < 0, "down", "up")

# add absolute amplitude column
waves$amplitude <- ifelse(waves$amplitude.male < 0, waves$amplitude.male * -1, waves$amplitude.male)

#------------------------PLOT ZONE----------------------------------------------------------------#

## Simple scatterplot of amplitude vs velocity
plot.av <- ggplot(waves, aes(x = velocity.male, y = amplitude, color = leg)) +
  geom_point() +
  geom_text(aes(label = X),hjust=-0.1, vjust=-0.1) +
  xlim(0, 300) +
  ylim(0, 45) +
  xlab("velocity (degrees/s)") +
  ylab("amplitude (degrees)") +
  ggtitle(
    exemplar,
    subtitle = paste0(
      "n waves: ",
      nrow(waves),
      ", mean amplitude: ",
      round(mean(waves$amplitude), digits = 2),
      ", mean velocity: ",
      round(mean(waves$velocity.male), digits = 2)
    )
  ) +
  theme_classic()

print(plot.av)
}