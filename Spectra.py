import numpy as np
import astropy.units as u
import multiprocessing as mp
import astropy.constants as c
import matplotlib.pyplot as plt
from astropy.coordinates import Distance
from SynchrotronSelfCompton import Synchrotron, SSC_Flux,I_sync,Int_Lum,SSC_cooling_rate,ElectronDistribution,tau_abs,tau_func


class SelfSynchrotronInverseCompton:
    def __init__(self, Dl, doppler, B, tvar, k, p, ebreak, gamma_2):
        ################################
        
        self.z = (Distance(Dl, u.Mpc)).compute_z()
        self.dl = (Dl*u.Mpc).to("cm")
        self.doppler = doppler
        self.time_var = tvar*u.s
        
        self.B = B*u.mG
        self.B_cgs = (self.B.to("G")).value*pow(u.g/u.cm,1.0/2.0)*pow(u.s,-1.0)
        self.B_crit = 4.414e13*u.G
        self.B_crit_cgs = 4.414e13*pow(u.g/u.cm,1.0/2.0)*pow(u.s,-1.0)
        
        self.U_B = pow(self.B_cgs,2.0)/(8*np.pi)
        self.e_B = (self.B/self.B_crit.to("mG"))
        self.u_B = ((self.U_B/(c.m_e.to("g")*pow(c.c.to("cm/s"),2.0)))).decompose()
        self.U_Bcrit = pow(self.B_crit_cgs, 2.0)/(8.0*np.pi)

        self.Lorentz_factor = 2.0*self.doppler
        self.beta = np.sqrt(1.0-1.0/pow(self.Lorentz_factor,2.0))
        
        self.Ke  = k
        self.p = p
        self.gamma_c = ebreak
        self.gamma_1 = 1.0e2
        self.gamma_2 = gamma_2
        self.xi = 10.0
        
        #################################

        self.cte_sync_delta = (pow(self.doppler,4.0)/(6.0*np.pi*pow(self.dl,2.0))*c.c.to("cm/s")*c.sigma_T.to("cm2")*self.U_B).to("erg/(cm^2 s)")
        self.cte_sync = ((np.sqrt(3.0)*pow(self.doppler,4)*pow(c.e.esu, 3)*self.B_cgs)/(4*np.pi*c.h.cgs*pow(self.dl,2.0))).to("erg/(cm2 s)")
        self.cte_x = ((4*np.pi*pow(c.m_e.to("g"),2.0)*pow(c.c.to("cm/s"),2))/((3.0*c.e.si*self.B.to("T")*c.h).to("g J"))).to("J/J")
        self.cte_SSC = (9.0/16.0)*((pow(1.0+self.z,2.0)*(c.sigma_T).to("cm2"))/(np.pi*pow(self.doppler,2.0)*(pow(c.c,2.0)).to("cm2/s2")*pow(self.time_var,2.0)))
        self.cte_W_par = (3*np.pi*pow(self.dl,2)*self.xi*(c.m_e).to("g")*pow(c.c.to("cm/s"),2))/((c.sigma_T).to("cm2")*c.c.to("cm/s")*pow(self.e_B,3./2.)*pow(1+self.z,1./2.)*self.U_Bcrit*pow(self.doppler,7./2.))
        self.epsilon_B_min_cte = (pow(3.0/2.0,2.0)*(self.xi*c.m_e.to("g")*pow(c.c.to("cm/s"),2.0)*pow(self.dl,2.0)*pow(1.0+self.z,5.0/2.0))/(2.0*pow(self.time_var,3.0)*pow(c.c.to("cm/s"),4.0)*c.sigma_T.to("cm2")*pow(self.U_Bcrit,2)*pow(self.doppler,13./2.))).to("s3/g")
        self.cte_u_sync = ((3.0*pow(self.dl,2.0)*pow(1.0+self.z,2.0))/(pow(c.c.to("cm/s"),3.0)*pow(self.time_var,2.0)*pow(self.doppler,6.0)))
        self.pair_opp_cte = (9.0*pow(self.dl,2.0)*c.sigma_T.to("cm2")*(1.0+self.z))/(8.0*c.m_e.to("g")*pow(c.c.to("cm/s"),4.0)*self.time_var*pow(self.doppler,5.0))
        ################################
        #print("B (cgs) = {:.3e}".format(self.B_cgs))
        #print("U_B = {:.5e}".format(self.U_B))
        #print("u_B = {:.5e}".format(self.u_B))
        #print("Synchrotron (delta) cte = {:.5e}".format(self.cte_sync_delta))
        #print("Synchrotron cte = {:.5e}".format(self.cte_sync))
        #print("x_cte = {:.5e}".format(self.cte_x))
        #print("Synchrotron-SelfCompton cte = {:.5e}".format(self.cte_SSC))
        #print("W_par_prime cte = {:.5e}".format(self.cte_W_par))
        #print("epsilon_b_min cte = {:.5e}".format(self.epsilon_B_min_cte))
        #print("u_prime(epsilon prime) cte = {:.5e}".format(self.cte_u_sync))
        #print("Photoabsorption cte = {:.5e}".format(self.pair_opp_cte))
    
    def Synchrotron(self, epsilon):
        return Synchrotron(epsilon, self.cte_sync.value, self.cte_x, self.z,  self.doppler, self.Ke,  self.p,  self.gamma_c,  self.gamma_1, self.gamma_2)


    def Synchrotron_Spectra(self, writeFile = False,  output_energy_units = "Hz", output_flux_units = "erg/(cm2 s)"):
        nu = np.logspace(8.5, 19, 50)
        A = (nu*u.Hz)*c.h
        B = c.m_e*pow(c.c,2)
        epsilon = A/(B.to("Hz J s"))
        sync_flux = map(self.Synchrotron, epsilon)
        sync_flux =list(sync_flux)

        sync_flux = sync_flux*u.erg*pow(u.cm,-2)*pow(u.s,-1)

        if output_energy_units != "Hz" or output_flux_units != "erg/(cm2 s)":
            if output_energy_units != "Hz":
                nu = A.to(output_energy_units).value
            if output_flux_units != "erg/(cm2 s)":
                sync_flux = sync_flux.to(output_flux_units)

        return nu, sync_flux.value

    def __photoabsortion__(self, epsilon_s):
        tau_gg = tau_abs(epsilon_s, self.pair_opp_cte.value,self.cte_sync_delta.value, self.z,self.doppler,self.e_B,self.Ke,self.p,self.gamma_c, self.gamma_1, self.gamma_2)
        #print("epsilon_s = {:.5e} | tau_gg = {:.5e}".format(epsilon_s,tau_gg))
        if tau_gg != 0.0:
            return (1.0-np.exp(-tau_gg))/tau_gg
        else:
            return 1.0
    
    def SSC(self, epsilon_s):
        absoprtion = self.__photoabsortion__(epsilon_s)
        tmp =  SSC_Flux(epsilon_s, self.cte_SSC.value, self.cte_sync.value, self.cte_x, self.z, self.doppler, self.Ke, self.p, self.gamma_c, self.gamma_1, self.gamma_2)*absoprtion
        print("epsilon s =  {:.5e} tmp = {:.5e}".format(epsilon_s, tmp))
        return tmp

    def SSC_Spectra(self, output_energy_units = "Hz", output_flux_units = "erg/(cm2 s)"):
        nu = np.logspace(16, 27.5, 40)
        A = (nu*u.Hz)*c.h
        B = c.m_e*pow(c.c,2)
        epsilon = A/(B.to("Hz J s"))

        pool = mp.Pool(processes=mp.cpu_count())
        ssc_flux = [pool.map_async(self.SSC,epsilon)]
        output = [p.get() for p in ssc_flux]
        pool.close()

        output[0] = output[0]*u.erg*pow(u.cm,-2)*pow(u.s,-1)

        if output_energy_units != "Hz" or output_flux_units != "erg/(cm2 s)":
            if output_energy_units != "Hz":
                nu = A.to(output_energy_units).value
            if output_flux_units != "erg/(cm2 s)":
                output[0] = output[0].to(output_flux_units)
        
        return nu, output[0].value
    


    ###########################################################################################################################################
    #
    # Constraining functions on the power jet
    #
    ###########################################################################################################################################

    def Blob_radius(self):
        '''
        Radius of the blob in the comovil frame. 
        Units are in cm
        '''
        return (c.c.to("cm/s")*self.doppler*self.time_var)/(1.0+self.z)

    def Energy_MagneticField(self):
        return ((pow(self.Blob_radius(),3.0)*pow(self.B_cgs,2.0))/6.0).to("erg")

    def Energy_particles(self):
        return (self.cte_W_par*I_sync(self.cte_sync_delta.value, self.z, self.doppler, self.e_B, self.Ke, self.p, self.gamma_c, self.gamma_1, self.gamma_2)*u.erg*pow(u.cm,-2.0)*pow(u.s,-1.0)).to("erg")

    def Power_jet(self):
        W_tot  = self.Energy_MagneticField() + self.Energy_particles()
        Blob_vol = 4./3.*np.pi*pow(self.Blob_radius(),3.0)
        return (2*np.pi*pow(self.Blob_radius(),2)*self.beta*pow(self.Lorentz_factor,2.0)*c.c.to("cm/s")*W_tot/Blob_vol).to("erg/s")

    def Epsilon_B_min(self):
        I = (I_sync(self.cte_sync_delta.value, self.z, self.doppler, self.e_B, self.Ke, self.p, self.gamma_c, self.gamma_1, self.gamma_2)*u.erg*pow(u.cm,-2)*pow(u.s,-1)).to("g cm2/(s2 cm2 s)")
        return pow(self.epsilon_B_min_cte*I,2./7.)

    def zeta_B(self):
        return self.e_B/self.Epsilon_B_min()

    def __Power_jet_min__(self):
        return ((14./3.)*np.pi*pow(self.vc,3.0)*pow((self.doppler*self.Lorentz_factor*self.time_var*self.Epsilon_B_min())/(1+self.z),2.0)*beta*self.U_Bcrit).to("erg/s")

    def PowerJet_Ratio(self):
        return (3./7.)*(pow(self.zeta_B(),2.)+(4./3.)*pow(self.zeta_B(),-3./2.))
    
    def Luminosity(self):
        return (2.0*np.pi*pow(self.dl,2.0)/pow(self.Lorentz_factor,2.0)*Int_Lum(self.cte_sync_delta.value, self.z, self.doppler, self.e_B, self.Ke, self.p, self.gamma_c, self.gamma_1, self.gamma_2)*u.erg*pow(u.cm,-2)*pow(u.s,-1)).to("erg/s")
        
    ###########################################################################################################################################
    #
    # Temporal variability
    #
    ###########################################################################################################################################


    def Lamor_Freq(self):
        return ((c.e.esu*self.B_cgs)/(2.0*np.pi*c.m_e.to("g")*c.c.to("cm/s"))).to("Hz")
    

    def t_acc_prime(self, gamma_prime, Na):
        return (Na*gamma_prime/(self.Lamor_Freq())).to("s")

    def Synchrotron_cool_rate(self, gamma_prime):
        return ((4.0/3.0)*c.c.to("cm/s")*c.sigma_T.to("cm2")*(self.U_B/(c.m_e.to("g")*pow(c.c.to("cm/s"),2)))*pow(gamma_prime,2)).to("Hz")

    def Synchrotron_cooling_time(self, gamma_prime):
        return (((1.0+self.z)/self.doppler)*(gamma_prime/self.Synchrotron_cool_rate(gamma_prime))).to("s")
        
        
    def SSC_cool_rate(self, gamma_prime):
        return (((3.0*c.sigma_T.to("cm2"))/(8.0*c.m_e.to("g")*c.c.to("cm/s")))*(SSC_cooling_rate( gamma_prime,  self.cte_u_sync.value,  self.cte_sync_delta.value,  self.z,  self.doppler,  self.e_B,  self.Ke,  self.p,  self.gamma_c,  self.gamma_1,  self.gamma_2)*u.erg*pow(u.cm,-2.0)*pow(u.cm,-1.0))).to("Hz")

    def SSC_cooling_time(self, gamma_prime):
        return (((1.0+self.z)/self.doppler)*(gamma_prime/self.SSC_cool_rate(gamma_prime))).to("s")


    ###########################################################################################################################################
    #
    # Electron distribution
    #
    ###########################################################################################################################################

    def ElectronDistribution(self, gamma_prime):
        return ElectronDistribution(gamma_prime , model.Ke, model.p, model.gamma_c, model.gamma_1, model.gamma_2)

if __name__== "__main__":

    def models(n):
        if n == 1:
            return "S06_1",  30, 282.00, 41.00, 1.10e+05, 1.30e+05, 9.00e+38
        elif n == 2:
            return "S06_2", 300, 278.00, 5.900, 1.10e+05, 4.80e+05, 6.00e+40
        elif n == 3:
            return "S06_3",3000, 168.00, 2.600, 2.10e+05, 9.40e+05, 3.00e+41
        elif n == 4:
            return "D07_1",  30, 230.00, 88.00, 3.10e+04, 1.30e+05, 2.00e+40
        elif n == 5:
            return "D07_2", 300, 124.00, 58.00, 5.20e+04, 2.20e+05, 5.00e+40
        elif n == 6:
            return "D07_3",3000, 67.000, 35.00, 9.20e+04, 4.00e+05, 9.00e+41
        elif n == 7:
            return "P05_1",  30, 199.00, 150.0, 2.60e+04, 1.10e+05, 3.00e+40
        elif n == 8:
            return "P05_2", 300, 107.00, 100.0, 4.30e+04, 1.80e+05, 2.00e+41
        elif n == 9:
            return "P05_3",3000, 58.000, 35.00, 7.60e+04, 3.20e+05, 9.00e+41
            
            
    Dl, p = 540, 2.7
    Gamma_prime = np.logspace(3,5.5,20)

    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\t{}".format("Model Name", "t_var,min", "delta", "B",  "zeta ", "P_jet x10e46" , "P_j/P_j,min", "L x10e42", "L/P_j","R_b"))
    
    for i in range(5,6):
        
        model_name, tvar, doppler, B,  gamma_break, gamma_2, Ke = models(i)
        model = SelfSynchrotronInverseCompton(Dl, doppler, B, tvar, Ke, p, gamma_break, gamma_2)
        print("{}\t\t{:.0f}\t\t{:.0f}\t{:.1f}\t{:.2f}\t{:.2f}\t\t{:.0f}\t\t{:.2f}\t\t{:.2e}\t{:.2f}".format(model_name, model.time_var.value, model.doppler, model.B.value, model.zeta_B(),model.Power_jet().value/1e46 ,model.PowerJet_Ratio(), model.Luminosity().value/1e42, model.Luminosity()/model.Power_jet(),model.Blob_radius().value/1e15))
        
        '''
        file = open("output/TimeScales_model_"+str(i)+".dat", "w")
        for g_prime in Gamma_prime:
            t_acc_10 = (1.0+model.z)/model.doppler*model.t_acc_prime(g_prime, 10.0)
            t_acc_1000 = (1.0+model.z)/model.doppler*model.t_acc_prime(g_prime, 1000.0)
            t_sync = model.Synchrotron_cooling_time(g_prime)
            t_SSC = model.SSC_cooling_time(g_prime)
            Ne = model.ElectronDistribution(g_prime)
            file.write("{:.5e}\t{:.5e}\t{:.5e}\t{:.5}\t{:.5e}\t{:.5e}\t{:.5e}\n".format(g_prime,t_acc_10.value,t_acc_1000.value,model.time_var.value,t_sync.value,t_SSC.value,Ne))
        
        file.close()
        '''
        

        ene_sync, sync = model.Synchrotron_Spectra()
        l = [ene_sync, sync]
        np.savetxt("output/Synchrotron_Spectra_Model"+model_name+".dat", list(map(list, zip(*l))), delimiter = "\t", fmt = '%1.5e')

        ene_ssc, ssc = model.SSC_Spectra()
        l = [ene_ssc, ssc]
        np.savetxt("output/SynchrotronSelfCompton_Spectra_Model"+model_name+".dat", list(map(list, zip(*l))), delimiter = "\t", fmt = '%1.5e')

#
#    '''
#    Dl, doppler, B, tvar, k, p, ebreak, gamma_2 = 540, 124, 58, 300, 5.0e40, 2.7, 5.2e4, 2.2e5
#    model = SelfSynchrotronInverseCompton(Dl, doppler, B, tvar, k, p, ebreak, gamma_2)
#
#    print("Blobs Radius = {:.2e}".format(model.Blob_radius()))
#    print("W_B_prime = {:.2e}".format(model.Energy_MagneticField()))
#    print("W_par_prime = {:.2e}".format(model.Energy_particles()))
#    print("Power jet = {:.2f} x10^46".format(model.Power_jet()/1e46))
#    print("epsilon_B,min = {:.2e}".format(model.Epsilon_B_min()))
#    print("zeta_B = {:.2f}".format(model.zeta_B()))
#    print("P_jet/P_jet,min = {:.2f}".format(model.PowerJet_Ratio()))
#    print("Total Luminosity = {:.2e}".format(model.Luminosity()))
#    print("Lamor Frequency = {:.2f}".format(model.Lamor_Freq()))
#    '''
