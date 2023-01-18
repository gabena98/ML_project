library(ggplot2)
library(tidyr)
library(dplyr)
library(RColorBrewer)

sl_cup_svm_2=read.csv("./grid_search_results/sl_cup_svm2_second.csv")

DF = data.frame(Models=c("sigma=2.5\nC=0.7","sigma=0.01\nC=5","sigma=2.5\nC=10",
                         "sigma=0.5\nC=0.1","sigma=0.2\nC=1000"),
                MSE=sl_cup_svm_2$sl_cup_svm_2.cvRisk[sl_cup_svm_2$sl_cup_svm_2.coef>0],
                Weights=sl_cup_svm_2$sl_cup_svm_2.coef[sl_cup_svm_2$sl_cup_svm_2.coef>0])

DFlong <- DF |> pivot_longer(cols = -Models,names_to = "Type") |> 
  mutate(scaled_value=ifelse(Type=="MSE",value,value*2))
#head(DFlong)
ggplot(DFlong,aes(x=Models, y = scaled_value, fill= Type)) + 
  geom_col(position="dodge") +
  geom_text(aes(label = round(value,4)), vjust = -0.5,
            position = position_dodge(width = 0.9))+
  scale_y_continuous(sec.axis = sec_axis(~(./2), name = "Weights")) +
  labs(y="MSE")
