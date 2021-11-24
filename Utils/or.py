from model import Perceptron
from all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np
#Or gate data

data={"x1":[0,0,1,1],"x2":[0,1,0,1],"y":[0,1,1,1]}
OR=pd.DataFrame(data)



x,y= prepare_data(OR)

ETA = 0.3 # 0 and 1
EPOCHS= 10
model_or= Perceptron(eta=ETA,epochs=EPOCHS)
model_or.fit(x,y)
_=model_or.total_loss()

save_model(model_or, filename="or.model")
save_plot(OR,"OR.png",model_or)
