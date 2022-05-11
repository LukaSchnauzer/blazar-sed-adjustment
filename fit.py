from datetime import datetime
from gradientDescent import *
from scipy.misc import derivative
from wrapper import *
import random

#B = 1e1 * a.Gauss

dL = 473.3

# Function to get two numpy arrays from the errorbars file
# Ea with the energies
# Fsyn with the fluxes
def getFromFile(file):
	Ea = np.empty(0)
	Fsyn = np.empty(0)
	read = open(file,"r")
	data = read.readlines()
	dataE_F = [] #list with only the energy and flux as string

	for i in range(0,len(data)):
		dataE_F.append(data[i][0:25])
		tab = dataE_F[i].index("\t")
		e = dataE_F[i][0:tab]
		f = dataE_F[i][tab+1:len(dataE_F[i])]
		Ea  = np.append(Ea,float(e))
		Fsyn = np.append(Fsyn,float(f))

	read.close()
	return Ea, Fsyn

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------- PARTIAL DEERIVATIVES FOR THE GRADIENT --------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Synchrotron:
def dFdoppler(x, params):
	a = (x*u.Hz)*c.h
	b = c.m_e*pow(c.c,2)
	epsilon = a/(b.to("Hz J s"))
	return derivative(lambda doppler: np.log10(Synchrotron(epsilon, dL, doppler ,params[1],10**params[2],params[3],10**params[4],10**params[5])), params[0], dx=0.5)

def dFB(x, params):
	a = (x*u.Hz)*c.h
	b = c.m_e*pow(c.c,2)
	epsilon = a/(b.to("Hz J s"))
	return derivative(lambda B: np.log10(Synchrotron(epsilon,dL,params[0], B ,10**params[2],params[3],10**params[4],10**params[5])), params[1], dx=0.5)

def dFKe(x, params):
	a = (x*u.Hz)*c.h
	b = c.m_e*pow(c.c,2)
	epsilon = a/(b.to("Hz J s"))
	return derivative(lambda Ke: np.log10(Synchrotron(epsilon,dL,params[0],params[1], Ke ,params[3],10**params[4],10**params[5])), 10**params[2], dx=0.5)

def dFp(x, params):
	a = (x*u.Hz)*c.h
	b = c.m_e*pow(c.c,2)
	epsilon = a/(b.to("Hz J s"))
	return derivative(lambda p: np.log10(Synchrotron(epsilon,dL,params[0],params[1],10**params[2], p ,10**params[4],10**params[5])), params[3], dx=0.5)

def dFgamma_c(x, params):
	a = (x*u.Hz)*c.h
	b = c.m_e*pow(c.c,2)
	epsilon = a/(b.to("Hz J s"))
	return derivative(lambda gamma_c: np.log10(Synchrotron(epsilon,dL,params[0],params[1],10**params[2],params[3], gamma_c ,10**params[5])), 10**params[4], dx=0.5)

def dFgamma_2(x, params):
	a = (x*u.Hz)*c.h
	b = c.m_e*pow(c.c,2)
	epsilon = a/(b.to("Hz J s"))
	return derivative(lambda gamma_2: np.log10(Synchrotron(epsilon,dL,params[0],params[1],10**params[2],params[3],10**params[4],gamma_2)), 10**params[5], dx=0.5)

def F(X,params):
	A = (X*u.Hz)*c.h
	B = c.m_e*pow(c.c,2)
	Epsilon = A/(B.to("Hz J s"))
	return np.log10(Synchrotron(Epsilon,dL,params[0],params[1],10**params[2],params[3],10**params[4],10**params[5]))

def generateRandomParameters():
	doppler = random.uniform(20.0,50.0)
	B = random.uniform(10.0,100.0)
	Ke = random.uniform(40.0, 45.0)
	p = random.uniform(2.0,3.1)
	gamma_c = random.uniform(4.5,5.6)
	gamma_2 = random.uniform(gamma_c,7.0)
	return np.array([doppler,B,Ke,p,gamma_c,gamma_2])

def generateRandomParametersSSC():
	return np.array([random.uniform(3.0,6.0)])

#Synchrotron
# datetime object containing current date and time
start = datetime.now()


files = ["WComaeDataSynch_Hz-erg.dat"]


X = np.array([])
Y = np.array([])

for f in files:
	temp = getFromFile(f)
	X = np.append(X,temp[0])
	Y = np.append(Y,temp[1])

#normalization value
Y = np.log10(Y)

#
initialParams = np.array([34.75278008, 85.85540838, 44.10540808, 2.25671054, 3.96448019, 5.54156217]) #generateRandomParameters() #np.array([34.75278008, 85.85540838, 44.10540808, 2.25671054, 3.96448019, 5.54156217]) # <-- this gives a ~0.0086 error  ~~~~    
derivatives = [dFdoppler,dFB,dFKe,dFp,dFgamma_c,dFgamma_2]
alpha = 1e-1
max_iter = 1
precision = 1e-6
condition = True
finalSyncParams = [0,0,0,0,0,0]
errorSync = 1e5
syncIterations = 0

results = ""

while condition:
	try:
		res = gradientDescent(X,Y,initialParams,alpha,F,derivatives,max_iter,precision,momentum = 0.9,beta=0.9)
		finalSyncParams = res[0]
		errorSync = res[1]
		syncIterations = res[2]
		results = results + ("\n"+res[3])
		condition = False
	except Exception as e:
		print(e)
		print ("\n-----Bad Parameters, starting over-----\n")
		results = results + "\n-----Bad Parameters, starting over-----\n"
		initialParams = generateRandomParameters()


#SSC:
#Datos obtenidos primero de la aproximacion Synchrotron
"""
dopplerSyn = 35.5
Bsyn = 85
Kesyn = 44.5
psyn = 2.25
gamma_csyn = 3.845
gamma_2syn = 5.477
"""
dopplerSyn = finalSyncParams[0]
Bsyn = finalSyncParams[1]
Kesyn = finalSyncParams[2]
psyn = finalSyncParams[3]
gamma_csyn = finalSyncParams[4]
gamma_2syn = finalSyncParams[5]

def dFTvar(x,params):
	a = (x*u.Hz)*c.h
	b = c.m_e*pow(c.c,2)
	epsilon_s = a/(b.to("Hz J s"))
	return derivative(lambda tvar: np.log10(SSC(epsilon_s, dL, dopplerSyn, Bsyn, tvar,10**Kesyn, psyn, 10**gamma_csyn, 10**gamma_2syn)), 10**params[0], dx=0.5)

def FSSC(X,params):
	A = (X*u.Hz)*c.h
	B = c.m_e*pow(c.c,2)
	EpsilonS = A/(B.to("Hz J s"))
	return np.log10(SSC(EpsilonS, dL, dopplerSyn, Bsyn, 10**params[0], 10**Kesyn, psyn, 10**gamma_csyn, 10**gamma_2syn))


files = ["WComaeDataSSC_Hz-erg.dat"]
X = np.array([])
Y = np.array([])

for f in files:
	temp = getFromFile(f)
	X = np.append(X,temp[0])
	Y = np.append(Y,temp[1])

#normalization value
Y = np.log10(Y)

initialParams = np.array([4.70690421]) #is the best param encountered yet ~~~~~~~~ generateRandomParametersSSC() 
derivatives = [dFTvar]
alpha = 5e-1
max_iter = 10000
precision = 1e-8
condition = True
finalSSCParams = [0]
errorSSC = 1e5
SSCIterations = 0

while condition:
	try:
		res= gradientDescent(X,Y,initialParams,alpha,FSSC,derivatives,max_iter,precision,momentum = 0.9,beta=0.9)
		finalSSCParams = res[0]
		errorSSC = res[1]
		SSCIterations = res[2]
		print(str(res[3]))
		results = results + ("\n"+res[3])
		condition = False
	except Exception as e:
		print(e)
		print ("\n-----Bad Parameters, starting over-----\n")
		results = results + "\n-----Bad Parameters, starting over-----\n"
		initialParams = generateRandomParametersSSC()

end = datetime.now()


results_tail = "\n----------Simulation Finished----------------\n"
results_tail =  results_tail + "Synchrotron params = "+ str(finalSyncParams)+ "\n"
results_tail =  results_tail + "Synchrotron error = "+ str(errorSync)+ "\n"
results_tail =  results_tail + "Iterations required = "+ str(syncIterations)+ "\n"
results_tail =  results_tail + "SSC params = "+ str(finalSSCParams)+ "\n"
results_tail =  results_tail + "SSC error = "+ str(errorSSC)+ "\n"
results_tail =  results_tail + "Iterations required = "+ str(SSCIterations)+ "\n"
results_tail =  results_tail + "From "+ str(start)+ " to "+ str(end)+ "\n"

print(results_tail)
results = results + results_tail

ResultsFile = open("resultadosPrueba.txt","w")
ResultsFile.write(str(results))
ResultsFile.close()