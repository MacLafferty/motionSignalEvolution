## calculate male head width. Takes raw_data.all from build_spider.r

coxa_left <- with(raw_data.all, data.frame(x = pt3_X,
                                           y = pt3_Y,
                                           z = pt3_Z))

coxa_right <- with(raw_data.all, data.frame(x = pt7_X,
                                            y = pt7_Y,
                                            z = pt7_Z))

cox_cox <- apply(coxa_left - coxa_right, 1, norm_vec)

cox_frames <- data.frame(cox = cox_cox, id = raw_data.all$id)

cox_id <- cox_frames %>% group_by(id) %>% summarise(mean_width = mean(cox),
                                                    sd_width = sd(cox),
                                                    n = n())

male <- cox_id %>% summarise(mean_width = mean(mean_width),
                             sd_width = mean(sd_width),
                             mean_frames = mean(n),
                             n = n())
