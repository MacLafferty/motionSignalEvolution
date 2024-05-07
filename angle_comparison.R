## script to visualize relationship of wave amplitude and distance
library(dplyr)
library(tidyr)
library(ggplot2)

source("R/build_spider_fun.R")

# Result: movement perceived by female is 2.32x higher than if waves were kept 
# static
size_static <- 1 
size_wave <- seq(2, 3.5, length.out = 36) # males increase amplitude 1.75 betw 10-45mm
distance <- c(10:45)
angle_static <- rad2deg(atan(size_static/distance))
angle_wave <- rad2deg(atan(size_wave/distance)) # atan(3 + 0.1 * x / x)

comparison <- data.frame(size_static,
                         size_wave,
                         distance,
                         angle_static,
                         angle_wave)

comparison <- gather(comparison, "type", "angle_moved", 4:5)

comp <- ggplot() +
  geom_line(data = comparison, aes(x = distance, y = angle_moved, color = type)) +
  scale_color_discrete(labels = c("static wave", "dynamic wave")) +
  theme(legend.title=element_blank()) +
  theme(legend.position = c(.75, .9)) +
  # theme(legend.position = "bottom") +
  xlim(0, 50) +
  #ylim(0, 20) +
  xlab("distance (mm)") +
  ylab("visual angle (degrees)")

my.ggsave("figures/static vs increasing wave.jpg", comp)

