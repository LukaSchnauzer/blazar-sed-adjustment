import numpy as np
import matplotlib.pyplot as plt

# Gradient Descent Step:
# X := array for the input (type array)
# Y := array for expected value (type array)
# F := callable function to aproximate data (type function)
# The function F must expect the input array value X and a list of parameters
# derivatives := list of partial derivatives over every parameter of F (type list of functions)
# The functions on the list must expect one input value x and a list of parameters
# and the variables over wich they are derivated must appear in the same order as in Theta_n
# Theta_n := parameters, array of parameters (type list)
# alpha := lerning rate (type float)
# max_iter := maximum iterations desired (tipe int)
# precision := desired precision (type float)
def gradientDescent(X,Y,Theta_n,alpha,F,derivatives,max_iter,precision,momentum = 0.0,beta = 0.5, forceLowError = False, errorThreshold = 100.0):
	S = "Starting Gradient Descent:\n"
	print(S)
	error = 1e5
	iters = 0
	gradientSum = np.zeros(Theta_n.size)
	epsilon = np.full(Theta_n.shape,1e-8)
	lastChange = np.zeros(Theta_n.size)

	for i in range(max_iter):
		iters+=1
		infoStr = "---------------Step "+str(i)+"-------------------\n"
		infoStr = infoStr + "Parameters : "+str(Theta_n)+"\n"
		currentTheta = Theta_n
		g,e,f,s_aux = gradE(X,Y,currentTheta,F,derivatives)

		infoStr = infoStr + s_aux

		#NORMAL (BATCH)
		#Theta_n = currentTheta - alpha*g

		#MOMENTUM
		#Theta_n = currentTheta - (alpha*g + momentum*lastChange)
		#print(alpha*g + momentum*lastChange)
		#lastChange = alpha*g + momentum*lastChange

		#ADAGRAD
		"""
		if(i<1): #On the first iteration we do a normal step
			lerningRate = np.full(Theta_n.shape,alpha)
		else:
			lerningRate = alpha / np.sqrt((epsilon + gradientSum))

		#print(lerningRate)
		Theta_n = currentTheta - lerningRate*g
		gradientSum = gradientSum + np.square(g)
		"""

		#RMSProp
		"""
		if(i<1): #On the first iteration we do a normal step
			lerningRate = np.full(Theta_n.shape,alpha)
		else:
			lerningRate = alpha / np.sqrt((epsilon + gradientSum))

		#print(lerningRate)
		Theta_n = currentTheta - lerningRate*g
		gradientSum = beta*gradientSum + (1-beta)*np.square(g)
		"""

		#Adam
		gradientSum = beta*gradientSum + (1-beta)*np.square(g)
		lastChange = momentum*lastChange + (1-momentum)*g

		#biased 1
		mt = lastChange/(1-pow(momentum,i+1))
		#biased 2
		vt = gradientSum/(1-pow(beta,i+1))

		lerningRate = alpha / np.sqrt((epsilon + vt))

		Theta_n = currentTheta - lerningRate*mt

		step = error - e
		
		if abs(step) <= precision:
			infoStr = infoStr + "Precision achieved"
			break

		error = e

		print(infoStr)
		S = S + infoStr
		if forceLowError and error >= errorThreshold:
			raise NameError('otravez')


	return [Theta_n,error,iters,S] 

def gradE(X,Y,Theta_n,F,derivatives):
	m = X.size + 0.0
	arrayF = np.zeros((X.shape[0],))
	for i in range(0,X.shape[0]):
		f = F(X[i],Theta_n)
		arrayF[i] = f

	error = Error(X,Y,Theta_n,arrayF)
	S = "Error: "+str(error)+"\n"
	return 1/m*np.dot(dF(X,Theta_n,derivatives),(arrayF-Y)),error,arrayF,S

def dF(X,Theta_n,derivatives):
	j = Theta_n.shape[0]
	m = X.shape[0]
	d = np.zeros((j,m))
	for i in range(j):
		for k in range(m):
			d[i,k] = derivatives[i](X[k],Theta_n)
	return d

def Error(X,Y,Theta_n,arrayF):
	m = X.shape[0]
	return np.sum(np.square(arrayF-Y))/(2.*m)