daniel <- read_csv("exemplars/2625_Daniel_waves.csv")
david <- read_csv("exemplars/2625_David_waves.csv")
isabella <- read_csv("exemplars/2625_Isabella_waves.csv")

daniel$scorer <- "daniel"
david$scorer <- "david"
isabella$scorer <- "isabella"
comp <- rbind(daniel, david, isabella)
