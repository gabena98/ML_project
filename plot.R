library(ggplot2)
library(tidyr)
library(dplyr)
library(RColorBrewer)

sl_cup_2_best=read.csv("./SL_finali/sl_cup_2_best.csv")

DF = data.frame(Models=c("Trees=250\nmtry=3","sigma=0.001\nC=3","sigma=3.5\nC=0.7","alpha=0.6\nnlambda=20"),
                MSE=sl_cup_2_best$sl_cup_2_best.cvRisk[sl_cup_2_best$sl_cup_2_best.coef>0],
                Weights=sl_cup_2_best$sl_cup_2_best.coef[sl_cup_2_best$sl_cup_2_best.coef>0])

DFlong <- DF |> pivot_longer(cols = -Models,names_to = "Type") |> 
  mutate(scaled_value=ifelse(Type=="MSE",value,value*2))
#head(DFlong)
ggplot(DFlong,aes(x=Models, y = scaled_value, fill= Type)) + 
  geom_col(position="dodge") +
  geom_text(aes(label = round(value,4)), vjust = -0.5,
            position = position_dodge(width = 0.9))+
  scale_y_continuous(sec.axis = sec_axis(~(./2), name = "Weights")) +
  labs(y="MSE")
