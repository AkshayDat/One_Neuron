from model import Perceptron
from all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np


#EXOR gate data

data={"x1":[0,0,1,1],"x2":[0,1,0,1],"y":[0,1,1,0]}
EXOR=pd.DataFrame(data)




x,y= prepare_data(EXOR)

ETA = 0.3 # 0 and 1
EPOCHS= 10
model_exor= Perceptron(eta=ETA,epochs=EPOCHS)
model_exor.fit(x,y)
_=model_exor.total_loss()

save_model(model_exor, filename="exor.model")
save_plot(EXOR,"EXOR.png",model_exor)
