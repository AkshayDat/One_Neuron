from Utils.model import Perceptron
from Utils.all_utils import prepare_data

#AND gate data

data={"x1":[0,0,1,1],"x2":[0,1,0,1],"y":[0,0,0,1]}
AND=pd.DataFrame(data)


x,y= prepare_data(AND)

ETA = 0.3 # 0 and 1
EPOCHS= 10
model= Perceptron(eta=ETA,epochs=EPOCHS)
model.fit(x,y)
_=model.total_loss()