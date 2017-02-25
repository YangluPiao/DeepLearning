import theano
from theano import tensor as T

#Initialize symbolic variables. Also could be vector, matrices, or even much higher demisonal tensors.
a=T.scalar()
b=T.scalar()

#Define symbolic expression
y=a*b

#Compile a function
multiply=theano.function(inputs=[a,b],outputs=y)

#Use the function on numeric data.
print (multiply(3,2)) #6
print (multiply(4,6)) #24