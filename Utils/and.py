from model import Perceptron
from all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np
#AND gate data

data={"x1":[0,0,1,1],"x2":[0,1,0,1],"y":[0,0,0,1]}
AND=pd.DataFrame(data)


x,y= prepare_data(AND)

ETA = 0.3 # 0 and 1
EPOCHS= 10
model= Perceptron(eta=ETA,epochs=EPOCHS)
model.fit(x,y)
_=model.total_loss()

save_model(model, filename="and.model")
save_plot(AND,"and.png",model)
