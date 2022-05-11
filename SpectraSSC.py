from SynchrotronSelfCompton import SynchrotronFlux, SSCFlux, Isync, ILum
import matplotlib.pyplot as plt
import astropy.constants as c
import multiprocessing as mp
import astropy.units as u
import numpy as np


G = pow(u.g, 1./2.)*pow(u.cm, -1./2.)*pow(u.s, -1.)
qe_cgs = 4.803e-10*pow(u.g, 1./2.)*pow(u.cm, 3./2.)*pow(u.s, -1.)
B_cr = 4.414e13*G
U_Bcr = pow(B_cr,2.)/(8*np.pi)
m_e_cgs = c.m_e.to("g")
c_cgs = c.c.to("cm/s")
h_cgs = c.h.to("cm2 g/s")
sigmaTcgs = c.sigma_T.to("cm2")


class SelfSynchrotronInverseCompton:

    def __init__(self, doppler, B, tvar, Ke, p, gamma_prime_break, gamma_prime_max, z, dL):

        self.doppler = doppler
        self.B = B*1e-3*G
        self.tvar = tvar*u.s
        
        self.z = z
        self.dL = (dL*u.Mpc).to("cm")
        
        
        #######################################################################################
        #
        # Electron distribution parameters
        #
        #######################################################################################
        
        self.Ke = Ke
        self.p = p
        self.gamma_prime_1 = 1e3
        self.gamma_prime_2 = gamma_prime_max
        
        self.gamma_prime_break = gamma_prime_break
        self.gamma_prime_max = gamma_prime_max
        
        
        #############################################
        # Constants on the code.
        #############################################
        
        self.e_B = self.B/B_cr
        self.xcons = (4.0*np.pi*pow(m_e_cgs,2.0)*pow(c_cgs,3.0))/(3.0*qe_cgs*self.B*h_cgs)
        self.SynCTE = ((np.sqrt(3.0)*pow(self.doppler, 4.0)*pow(qe_cgs,3.)*self.B)/(4*np.pi*h_cgs*pow(self.dL, 2.))).to("erg/(cm2 s)")
        self.SSCCTE = 9./16.*((pow(1.+self.z,2.)*sigmaTcgs)/(np.pi*pow(self.doppler,2.)*pow(c_cgs,2.)*pow(self.tvar,2.)))
        
        self.U_B = pow(self.B,2.)/(8.*np.pi)
        self.SyncDeltaCTE = (pow(self.doppler,4.)/(6*np.pi*pow(self.dL,2.))*c_cgs*sigmaTcgs*self.U_B).to("erg/(cm2 s)")
        
        self.xi = 10.0
        self.LF = doppler*0.9
        self.beta = pow(1. - 1./pow(self.LF,2.), 1./2.)
    
    
        #print("SSC Cte (delta) = {:.3e}".format(self.SyncDeltaCTE))
    
    ###############################################################################################
    #     Spectra functions
    ###############################################################################################
    
    
    def __synchrotron__(self, epsilon):
        return SynchrotronFlux(epsilon, self.Ke, self.p, self.gamma_prime_break, self.gamma_prime_1, self.gamma_prime_2, self.SynCTE.value ,self.xcons, self.z, self.doppler, self.gamma_prime_1, self.gamma_prime_2)
        
    def SynchrotronSpectra(self):
        epsilon = ((np.logspace(8.5, 20, 50)*u.Hz*c.h)/(c.m_e*pow(c.c,2))).decompose()
        sync_flux = np.array(list(map(self.__synchrotron__, epsilon)))
        return np.logspace(8.5, 20, 50), sync_flux
        
    def __selfSynchrotronCompton__(self, epsilon_s):
        return SSCFlux(epsilon_s , self.Ke, self.p, self.gamma_prime_break, self.gamma_prime_1, self.gamma_prime_2, self.SSCCTE, self.SynCTE.value, self.xcons.value, self.e_B, self.z, self.doppler)
        
    def SSCSpectra(self):
        epsilon = ((np.logspace(15, 27.46, 50)*u.Hz*c.h)/(c.m_e*pow(c.c,2))).decompose()
        
        pool = mp.Pool(processes=mp.cpu_count())
        ssc_flux = [pool.map_async(self.__selfSynchrotronCompton__,epsilon)]
        output = [p.get() for p in ssc_flux]
        pool.close()
        
        return np.logspace(15, 27.46, 50),output[0]

    ###############################################################################################
    #     Constrains in jet parameters and magnetic field, derived quantities.
    ###############################################################################################
    
    
    def R_prime_blob(self):
        '''
        Comoving radius of the emitting blob
        '''
        return (self.tvar*self.doppler*c_cgs)/(1.+self.z)
        
    def V_prime_blob(self):
        '''
        Comoving volumen of the blob
        '''
        return 4.*np.pi*pow(self.R_prime_blob(),3.)/3.
        
    def W_prime_B(self):
        '''
        Comoving energy in the magnetic field
        '''
        return (self.V_prime_blob()*self.U_B).to("erg")
        
    def W_prime_e(self):
        '''
        Total comoving energy in the electrons
        '''
        cteWe = (m_e_cgs*pow(c_cgs,2.) *(6.*np.pi*pow(self.dL,2.))/(c_cgs*sigmaTcgs*pow(self.e_B ,2.)*U_Bcr*pow(self.doppler,4.))*(1./2.)*np.sqrt((self.doppler*self.e_B)/(1.+z))).to("cm2 s")
        I = Isync(self.SyncDeltaCTE.value, self.Ke, self.p, self.gamma_prime_break, self.gamma_prime_1, self.gamma_prime_2, self.e_B, self.z, self.doppler)*u.erg*pow(u.cm,-2.)*pow(u.s,-1.)
        return (cteWe*I).to("erg")
    
    def W_prime_particle(self):
        '''
        Total energy in electrons and protons:
        '''
        return self.xi*self.W_prime_e()
        
        
    def Pj(self):
        '''
        Total power jet in the stationary frame (i.e. the frame of the galaxy)
        '''
        Wtot = self.W_prime_particle() + self.W_prime_B()
        return 2*np.pi*pow(self.R_prime_blob(),2.)*self.beta*pow(self.LF,2.)*c_cgs*(Wtot/self.V_prime_blob())
        
        
    def e_B_min(self):
        '''
        Magnetic field that minimize the power jet.
        '''
        
        I = Isync(self.SyncDeltaCTE.value, self.Ke, self.p, self.gamma_prime_break, self.gamma_prime_1, self.gamma_prime_2, self.e_B, self.z, self.doppler)*u.erg*pow(u.cm,-2.)*pow(u.s,-1.)
        #print("---------> Isync = {:.3e}".format(I))
        eBmin_cons = pow(3./2.,3.) * ((self.xi*m_e_cgs*pow(c_cgs,2.)*pow(self.dL, 2.)*pow(1.+self.z, 5./2.))/(2.*pow(self.tvar,3.)*pow(c_cgs,4.)*sigmaTcgs*pow(U_Bcr,2.)*pow(self.doppler, 13./2.)))

        return pow(eBmin_cons*I, 2./7. )
        
    def RatioMagneticFields(self):
        '''
        '''
        return self.e_B/self.e_B_min()
        
    def MiminumPowerJet(self):
        '''
        '''
        return ((14./3.)*np.pi*pow(c_cgs,3.)*pow( (self.doppler*self.LF*self.tvar*self.e_B_min())/(1. + self.z) ,2.)*self.beta*U_Bcr).to("erg/s")
        
    def JetPowerRatio(self):
        zeta_B = self.RatioMagneticFields()
        return (3./7.)*(pow(zeta_B,2.) + (4./3.)*pow(zeta_B, -3./2.))

    def TotalLuminosity(self):
        '''
        Total luminosity
        '''
        I = ILum(self.SyncDeltaCTE.value, self.Ke, self.p, self.gamma_prime_break, self.gamma_prime_1, self.gamma_prime_2, self.e_B, self.z, self.doppler)*u.erg*pow(u.cm,-2.)*pow(u.s,-1.)
        return (2.0*np.pi*pow(self.dL, 2.)/pow(self.LF, 2.)*I).to("erg/s")

    
    def printTable(self, model_name):
        print("{}\t\t{:.0f}\t\t{:.0f}\t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.0f}\t\t{:.2f}\t\t{:.2e}\t\t{:.2f}".format(model_name, self.tvar.value, self.doppler, self.B.value*1e3, self.RatioMagneticFields(),self.Pj()/1e46 ,self.JetPowerRatio(), self.TotalLuminosity()/1e42, self.TotalLuminosity()/self.Pj(),self.R_prime_blob()/1e15))


def models(n):
    if n == 1:
        return "S06_1",  30, 282.00, 41.00, 1.10e+05, 1.30e+05, 9.00e+38
    elif n == 2:
        return "S06_2", 300, 278.00, 5.900, 1.10e+05, 4.80e+05, 6.00e+40
    elif n == 3:
        return "S06_3",3000, 168.00, 2.600, 2.10e+05, 9.40e+05, 3.00e+41
    elif n == 4:
        return "D07_1",  30, 230.00, 88.00, 3.10e+04, 1.30e+05, 2.00e+40*1.2
    elif n == 5:
        return "D07_2", 300, 124.00, 58.00, 5.20e+04, 2.20e+05, 5.00e+40*2.9
    elif n == 6:
        return "D07_3",3000, 67.000, 35.00, 9.20e+04, 4.00e+05, 9.00e+41
    elif n == 7:
        return "P05_1",  30, 199.00, 150.0, 2.60e+04, 1.10e+05, 3.00e+40
    elif n == 8:
        return "P05_2", 300, 107.00, 100.0, 4.30e+04, 1.80e+05, 2.00e+41
    elif n == 9:
        return "P05_3",3000, 58.000, 35.00, 7.60e+04, 3.20e+05, 9.00e+41


if __name__== "__main__":
    z, dL, p = 0.116, 540, 2.7
    
    print("{}\t\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\t{}".format("Name", "t_v,min (s)", "delta", "B (mG)", "zeta_B", "P_j x10^42 (erg/s)" , "P_j/P_j,min", "L_T x10^42 (erg/s)", "L_T/P_j","R'_b x10^15 (cm)"))
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")
    for i in range(1,10):
        model_name, tvar, doppler, B,  gamma_prime_break, gamma_prime_max, Ke = models(i)
        model1 = SelfSynchrotronInverseCompton(doppler, B, tvar, Ke, p, gamma_prime_break, gamma_prime_max, z, dL)
        model1.printTable(model_name)
    '''
    plt.figure()
    plt.loglog(nu1, flux1, "r-")
    plt.loglog(nu12, flux12, "r-")
    
    plt.loglog(nu2, flux2,"b--")
    plt.loglog(nu22, flux22, "b--")
        
    plt.loglog(nu3, flux3,"g:")
    plt.loglog(nu32, flux32, "g:")
    
    plt.xlim(1e10, 1e29)
    plt.ylim(1e-12, 4e-9)
    plt.xlabel(r"$\nu$ (Hz)")
    plt.ylabel(r"$\nu f_{\nu}$ (erg cm$^{-2}$ s$^{-1}$)")
    plt.show()
    '''
    
