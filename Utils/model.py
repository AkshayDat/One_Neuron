class Perceptron:
    def __init__(self, eta, epochs):
        self.weights=np.random.randn(3) * 1e-4 #Small Weight INIT
        print("Initail Weights before training: {self.weight}")
        self.eta=eta #Learning Rate
        self.epochs=epochs
        #self.activationfun=activationfun
        
    def activation_function(self, inputs, weights):
        z = np.dot(inputs, weights)
        return np.where(z > 0, 1, 0)
        
        
        
    def fit(self,x,y):
        self.x=x
        self.y=y
        
        x_with_bias=np.c_[x,-np.ones((len(self.x),1))] # Concatenation
        print(f"X with bias: {x_with_bias}")
        
        for epochs in range(self.epochs):
            print("--"*10)
            print(f"for epoch: {epochs}")
            print("--"*10)
            
            
            y_hat=self.activation_function(x_with_bias,self.weights) # Forward Propogation
            print(f"Predicted value after forward Pass: {y_hat}")
            self.error=self.y - y_hat
            self.weights=self.weights+ self.eta* np.dot(x_with_bias.T,self.error)# Backward Propogation
            print(f"Updated weightrs after epoch: {epochs}/{self.epochs}")
            print("####"*10)
            
            
    def predict(self,x):
        x_with_bias=np.c_[x,-np.ones((len(x),1))]
        return self.activation_function(x_with_bias,self.weights)
    
    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"total loss: {total_loss}")
        return total_loss