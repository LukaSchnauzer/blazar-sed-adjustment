import numpy as np
import astropy.units as u
import multiprocessing as mp
import astropy.constants as c
from astropy.coordinates import Distance
from SynchrotronSelfCompton import SynchrotronFlux as SynchrotronImpl
from SynchrotronSelfCompton import SSCFlux as SSC_FluxImpl

G = pow(u.g, 1./2.)*pow(u.cm, -1./2.)*pow(u.s, -1.)
qe_cgs = 4.803e-10*pow(u.g, 1./2.)*pow(u.cm, 3./2.)*pow(u.s, -1.)
B_cr = 4.414e13*G
U_Bcr = pow(B_cr,2.)/(8*np.pi)
m_e_cgs = c.m_e.to("g")
c_cgs = c.c.to("cm/s")
h_cgs = c.h.to("cm2 g/s")
sigmaTcgs = c.sigma_T.to("cm2")

# wrapper funtion for easy partial derivation of parameters
def Synchrotron(epsilon, dL, doppler, B, Ke, p, gamma_c, gamma_2):
	z = 0.09959805845252324
	B = B*1e-3*G
	dL = (dL*u.Mpc).to("cm")
	gamma_1 = 1e2
	SynCTE = ((np.sqrt(3.0)*pow(doppler, 4.0)*pow(qe_cgs,3.)*B)/(4*np.pi*h_cgs*pow(dL, 2.))).to("erg/(cm2 s)").value
	xcons = (4.0*np.pi*pow(m_e_cgs,2.0)*pow(c_cgs,3.0))/(3.0*qe_cgs*B*h_cgs)
	return SynchrotronImpl(epsilon, Ke, p, gamma_c, gamma_1, gamma_2, SynCTE , xcons, z, doppler, gamma_1, gamma_2)

def SSC(epsilon_s, dL, doppler, B, tvar, Ke, p, gamma_c, gamma_2):
	z = 0.09959805845252324
	B = B*1e-3*G
	dL = (dL*u.Mpc).to("cm")
	gamma_1 = 1e2
	SynCTE = ((np.sqrt(3.0)*pow(doppler, 4.0)*pow(qe_cgs,3.)*B)/(4*np.pi*h_cgs*pow(dL, 2.))).to("erg/(cm2 s)").value
	xcons = (4.0*np.pi*pow(m_e_cgs,2.0)*pow(c_cgs,3.0))/(3.0*qe_cgs*B*h_cgs)
	e_B = B/B_cr
	SSCCTE = 9./16.*((pow(1.+z,2.)*sigmaTcgs)/(np.pi*pow(doppler,2.)*pow(c_cgs,2.)*pow(tvar,2.))).value
	return SSC_FluxImpl(epsilon_s, Ke, p, gamma_c, gamma_1, gamma_2, SSCCTE, SynCTE, xcons, e_B, z, doppler) 


#BEST:
"""
dopplerX = 34.75278008
BX = 85.85540838
KeX = 44.10540808
pX = 2.25671054
gamma_cX = 3.96448019
gamma_2X = 5.54156217
tvarX = 3.71436531
"""
dopplerX,BX,KeX,pX,gamma_cX,gamma_2X,tvarX = (34.01884133,84.14296321,44.10540808,2.51561181,3.96856286,5.54271467,3.7861565)
#Synchrotron test
nu = np.logspace(9.5, 19, 20)
A = (nu*u.Hz)*c.h
B = c.m_e*pow(c.c,2)
Epsilon = A/(B.to("Hz J s"))
sync_flux = [Synchrotron(epsilon.decompose(),473.3 , dopplerX, BX, 10**KeX, pX, 10**gamma_cX, 10**gamma_2X) for epsilon in Epsilon]
l = [nu, sync_flux]
np.savetxt("Sync_Spectra.dat", list(map(list, zip(*l))), delimiter = "\t", fmt = '%1.5e')
#SSC test
"""
nu = np.logspace(18.0, 28.15, 30)
A = (nu*u.Hz)*c.h
B = c.m_e*pow(c.c,2)
EpsilonS = A/(B.to("Hz J s"))
#pool = mp.Pool(processes=4)
#ssc_flux = [pool.map_async(SSC,epsilon)]
#output = [p.get() for p in ssc_flux]
#l = [nu, output[0]]
# Tvar = 7500
ssc_flux = [SSC(epsilon_s.decompose(),473.3 , dopplerX, BX,10**tvarX, 10**KeX, pX, 10**gamma_cX, 10**gamma_2X) for epsilon_s in EpsilonS]
l = [nu, ssc_flux]
np.savetxt("SSC_Spectra.dat", list(map(list, zip(*l))), delimiter = "\t", fmt = '%1.5e')
"""