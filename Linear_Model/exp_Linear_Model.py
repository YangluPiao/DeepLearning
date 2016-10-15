import theano
from theano import tensor as T
import numpy as np
import matplotlib  
import matplotlib.pyplot as plt 

#traing set.
train_x=np.linspace(-1,1,101)
train_y=2*train_x+np.random.randn(*train_x.shape)*0.33

#plot the points.
plt.figure()
plt.scatter(train_x,train_y)
plt.hold

#Initialize symbolic variables
X=T.scalar()
Y=T.scalar()

#Define symbolic model
def model(X,W):
    return X*W

#Initialize model parameter. "theano.shared" is a hybrid value. W is randomly initialized to 0.
W=theano.shared(np.asarray(0.,dtype=theano.config.floatX))
y=model(X,W)

#Define symbolic loss, cost function.
cost=T.mean(T.sqr(y-Y))

#gradient descent. Determine partial derivative of loss w.r.t parameter.
gradient = T.grad(cost=cost,wrt=W)
#Define how to update parameter based on gradient.
updates = [[W,W-gradient*0.01]]

#Compile trainging funtion. Result is value of cost function and W.
train=theano.function(inputs=[X,Y],outputs=[cost,W],updates=updates,allow_input_downcast=True)



#Iterate through data 100 times, updating parameter after each iteration.
for i in range(100):
    for x, y in zip(train_x,train_y):
        [J,theta]=train(x,y)

#Draw the regressed line.
result_y=train_x*theta
plt.plot(train_x,result_y,'r')