import numpy
import pickle
import bisect

from scipy.constants import G as G
from scipy.constants import Boltzmann as kb
import datetime
import matplotlib.pyplot as plt
import scipy.integrate

import pygpi5
import pygpi5.Models.MSIS as MSIS

class G4EPP:
    """
    Implements the Berland+2023 G4EPP model (https://doi.org/10.1029/2023EA002987). 
    """
    def __init__(self):
        self.Earray  = numpy.round(numpy.logspace(numpy.log10(0.250), numpy.log10(10000), 100), 2)
        self.h       = numpy.arange(0, 500);
        self.runList = numpy.array([10, 14, 20, 32, 50, 71, 100, 141, 200, 316, 500, 
                            707, 1000, 1414, 2000, 3162, 5000, 7071, 10000])
        self.PAlist  = numpy.arange(0, 70+5, 5);

        self.data_path = pygpi5.model_dir / "G4data_mono_discretePAD_0degLat.pkl"
        # TODO: Implement automatic download if file is not found. Ask for permission before downloading
        # https://zenodo.org/records/8034275/files/G4data_mono_discretePAD_0degLat.pkl?download=1
        assert self.data_path.exists(), (f"G4EPP pickle file not found in {self.data_path}. "
                                         "Download here: https://zenodo.org/records/8034275")
        self.D = pickle.load(open(self.data_path, "rb"))

        self.internal_weights = None
        self.internal_flux    = None


    def set_internal_weights(self, E, Q):
        returnArray = numpy.zeros(len(self.runList))
        
        # Find where queried energy fits within control energies
        left_ind = bisect.bisect_left(self.runList, E)
        
        if left_ind == 0: # left edge case
            returnArray[0] = 1.0

        elif left_ind >= len(self.runList) - 1: # right edge case
            returnArray[-1] = 1.0

        else: # nominal case
            # Queried energy lies within this interval, 
            # and find logarithmic percentage between the two points
            log_weight = numpy.log (E / self.runList[left_ind-1]) /\
                    numpy.log( self.runList[left_ind] / self.runList[left_ind-1] )

            # Set two adjacent beams to their relative weights
            returnArray[left_ind-1]   = 1 - log_weight
            returnArray[left_ind] = log_weight

        # Set the member variables
        self.internal_weights = returnArray
        self.internal_flux    = Q / E / 2

    def get_internal_weights(self):
        if self.internal_weights is None:
            raise ValueError("Did not set internal weights in G4EPP!")
        else:
            return self.internal_weights

    def get_internal_flux(self):
        if self.internal_flux is None:
            raise ValueError("Did not set internal flux in G4EPP!")
        else:
            return self.internal_flux

    def formGreensFunctionSpectrum(
            self,
            energyDistribution,
            pitchAngleDistribution,
            flux):

        particle = None
        dataType = 'ioni'

        testArray = numpy.hstack([energyDistribution, pitchAngleDistribution])

        if (numpy.isnan(testArray)).any():
            raise ValueError("Nan(s) in inputs!")

        if (numpy.isinf(testArray)).any():
            raise ValueError("Inf(s) in inputs!")

        # Energy array in eV for convienience
        energyAbsc = self.runList * 1e3

        # Normalize energy distribution
        energyDistribution     /= numpy.trapz(energyDistribution, x=energyAbsc)

        # Normalize pitch angle distribution
        pitchAngleDistribution /= numpy.trapz(pitchAngleDistribution, x=numpy.deg2rad(self.PAlist))


        # Compute energy flux Input 
        # Solid angle calculation
        int1 = 2 * numpy.pi * numpy.trapz(pitchAngleDistribution * numpy.sin(numpy.deg2rad(self.PAlist)),
                                    x=numpy.deg2rad(self.PAlist))

        # First moment of energy distribution
        int2 = numpy.trapz(energyDistribution * energyAbsc,
                        x=energyAbsc)

        norm = flux * int1 * int2

        data = self.D 

        if dataType == 'ioni':
            result     = numpy.zeros(500)
            multFactor = 0.035

        elif dataType == "spectra":
            result     = numpy.zeros([500, 100])
            multFactor = 1

        # TODO: make this a matrix multiplication for SPEED
        # See documentation for explanation of divisive factors in below loop
        for ind1, ene in enumerate(self.runList):
            for ind2, pa in enumerate(self.PAlist):

                weight = energyDistribution[ind1] * pitchAngleDistribution[ind2]

                angFactor = (1e5 *  2 * numpy.pi * numpy.cos(numpy.deg2rad(pa)) * multFactor)

                if particle is None:

                    result += weight * (data[('electron', dataType, ene, pa)][0] + \
                           data[('photon', dataType, ene, pa)][0]/100) / angFactor

                if particle == 'electron':

                    result += weight * data[('electron', dataType, ene, pa)][0] / angFactor

                if particle == 'photon':

                    result += weight * data[('photon', dataType, ene, pa)][0]/100 / angFactor


        # (ioni ~ cm^-3 s^-1, spectra ~ keV cm^-2 s^-1 sr^-1 keV^-1)
        # norm ~ eV / cm^2 / sec
        return result, norm


    def get_ionization_profile(self, altkm):
       
        energy_weights = self.get_internal_weights()
        flux           = self.get_internal_flux()

        result, norm = self.formGreensFunctionSpectrum(
                                    energy_weights,
                                    numpy.sin(numpy.deg2rad(self.PAlist)),  # sin PAD
                                    flux)

        # Normalize to unity
        result /= numpy.trapz(result, x=1e5 * self.h)
        
        # Make it integrate to what it should be
        result *= (norm/35)

        interp_result = 10**numpy.interp(altkm, self.h, numpy.log10(result), left=-numpy.inf, right=-numpy.inf)

        return interp_result


class Ionization:
    """

    """
    def __init__(self):
        """
        Implementing Fang et al., 2010
        """
        self.msis = MSIS.MSIS()
        self.Pij = numpy.zeros([7,4])
        self.Pij = numpy.array([[1.24616e0,1.45903e0, -2.42269e-1, 5.95459e-2], \
                                [2.23976e0, -4.22918e-7, 1.36458e-2, 2.53332e-3], \
                                [1.41754e0, 1.44597e-1, 1.70433e-2, 6.39717e-4],\
                                [2.48775e-1,-1.50890e-1, 6.30894e-9, 1.23707e-3], \
                                [-4.65119e-1, -1.05081e-1, -8.95701e-2, 1.22450e-2], \
                                [3.86019e-1, 1.75430e-3, -7.42960e-4, 4.60881e-4], \
                                [-6.45454e-1, 8.49555e-4, -4.28581e-2, -2.99302e-3], \
                                [9.48930e-1, 1.97385e-1, -2.50660e-3, -2.06938e-3]])
        self.PijMax = numpy.zeros([7,4])
        self.PijMax = numpy.array([[3.49979e-1,-6.18200e-2, -4.08124e-2, 1.65414e-2], \
                                [5.85425e-1, -5.00793e-2, 5.69309e-2, -4.02491e-3], \
                                [1.69692e-1, -2.58981e-2, 1.96822e-2, 1.20505e-3],\
                                [-1.2227e-1,-1.15532e-2, 5.37951e-6, 1.20189e-3], \
                                [1.57018e0, 2.87896e-1, -4.14857e-1, 5.18158e-2], \
                                [8.83195e-1, 4.31402e-2, -8.33599e-2, 1.02515e-2], \
                                [1.90953e0, -4.74704e-2, -1.80200e-1, 2.46652e-2], \
                                [-1.29566e0, -2.10952e-1, 2.73106e-1, -2.92752e-2]])
        
        return

    def GravitationalAcceleration(self, z):
        """
        Input Altitude in meters!
        G = m^3*kg^-1*s^-2
        units of meters
        """
        Me = 5.97237e24 # kg wikipedia source
        Re = 6.371e6 # m

        # probably good enough for altitude
        g = G*Me/(Re+z)**2

        return g


    def CalculateCi(self, E0):
        """
        Fang equation 5
        """
        Ci = numpy.zeros(8)
        x = numpy.log(E0)
        for i in range(8):
            y = self.Pij[i,0]+self.Pij[i,1]*x+self.Pij[i,2]*x**2+self.Pij[i,3]*x**3
            Ci[i] = numpy.exp(y)
        return Ci

    def CalculateCiMaxwellian(self, E0):
        """
        Fang equation 5
        """
        Ci = numpy.zeros(8)
        x = numpy.log(E0)
        for i in range(8):
            y = self.PijMax[i,0]+self.PijMax[i,1]*x+self.PijMax[i,2]*x**2+self.PijMax[i,3]*x**3
            Ci[i] = numpy.exp(y)
        return Ci

    def FangModel(self, E0,Q0, altkm, Hz,MassDensity):
        """
        Input:  E = MonoEnergetic Energy (eV)
                altkm = Altitude (km)
                Hz (m)
                Mass Density (kg/m^3)
        """
        # do all of the conversions here
        zcm = altkm*(1000.*100.) # cm
        EkeV = E0/1000. # KeV
        rhoz = MassDensity*(1e3/1e6) # g/cm^3
        Hzcm = Hz*100.
        deltaE = 0.035 #keV
        Q0 = Q0*6.242e11/1000. # conversion to keV/cm2/s

        y = (2./EkeV)*(rhoz*Hzcm/6e-6)**0.7 # equation 1
        Ci = self.CalculateCi(EkeV)
        f = Ci[0]*(y**Ci[1])*numpy.exp(-Ci[2]*(y**Ci[3]))+\
            Ci[4]*(y**Ci[5])*numpy.exp(-Ci[6]*(y**Ci[7])) # equation 4

        qz = f*Q0/deltaE/Hzcm # equation 3 which is wrong, look in fang 2008, eq2
        return qz, y, f

    def BerlandModel(self, E0, Q0, altkm, Hz, MassDensity):
        """
        GB new function here
        Input:  E = MonoEnergetic Energy (eV)
                altkm = Altitude (km)
                Hz (m)
                Mass Density (kg/m^3)
        """

        # do all of the conversions here
        zcm = altkm * (1000.0 * 100.0) # cm
        EkeV = E0 / 1000. # keV
        rhoz = MassDensity * ( 1e3 / 1e6 ) # g / cm^3
        Hzcm = Hz * 100.0
        deltaE = 0.035 # keV
        Q0 = Q0 * 6.242e11 / 1000.0 # conversion to keV / cm2 / s

        berland_model = G4EPP()
        berland_model.set_internal_weights(EkeV, Q0)
        result = berland_model.get_ionization_profile(altkm)
        return result 

    def FangModelMaxwellian(self, Q0,E0, altkm, Hz,MassDensity):
        """
        Input:  E = MonoEnergetic Energy (eV)
                altkm = Altitude (km)
                Hz (m)
                Mass Density (kg/m^3)
            Fang 2008 paper
        """
        # do all of the conversions here
        zcm = altkm*(1000.*100.) # cm
        EkeV = E0/1000. # KeV
        rhoz = MassDensity*(1e3/1e6) # g/cm^3
        Hzcm = Hz*100.
        deltaE = 0.035 #keV
        # Q0 = Q0/1000.
        Q0 = Q0*6.242e11/1000. # conversion to keV/cm2/s

        y = (1./EkeV)*(rhoz*Hzcm/4e-6)**0.606 # equation 4
        Ci = self.CalculateCiMaxwellian(EkeV)
        f = Ci[0]*(y**Ci[1])*numpy.exp(-Ci[2]*(y**Ci[3]))+\
            Ci[4]*(y**Ci[5])*numpy.exp(-Ci[6]*(y**Ci[7])) # equation 4

        qz = (Q0*f)/(2.*deltaE*Hzcm) # equation 3 which is wrong, look in fang 2008, eq2
        return qz, y, f


    def FangModelMatrix(self, Q0,E0, altkm, Hz,MassDensity):
        """
        Input:  E = MonoEnergetic Energy (eV)
                altkm = Altitude (km)
                Hz (m)
                Mass Density (kg/m^3)
        """
        # do all of the conversions here
        zcm = altkm*(1000.*100.) # cm
        EkeV = E0/1000. # KeV
        rhoz = MassDensity*(1e3/1e6) # g/cm^3
        Hzcm = Hz*100.
        deltaE = 0.035 #keV
        Q0 = Q0*6.242e11/1000. # conversion to keV/cm2/s
       
        y = (2./EkeV)*(rhoz*Hzcm/6e-6)**0.7 # equation 1
        Ci = self.CalculateCi(EkeV)
        f = Ci[0]*(y**Ci[1])*numpy.exp(-Ci[2]*(y**Ci[3]))+\
            Ci[4]*(y**Ci[5])*numpy.exp(-Ci[6]*(y**Ci[7])) # equation 4

        qz = f*Q0/deltaE/Hzcm # equation 3 which is wrong, look in fang 2008, eq2
        return qz, y, f

    def RunMSISFang(self, tUnix, glat,glon,altkm = numpy.arange(80,150,1)):
        """
        This should only have to be run once to calculate what I need
        Input: unix time, longitude and latitude, altitude grid
        Do everything in SI
        """
        year = int(tUnix/(24.*3600.*365)+1970.)
        doy = int((tUnix/(24.*3600))%365)
        utHrs = (tUnix/3600.)%24
        CGSorSI = 'SI'
        outDictSI = self.msis.MSIS(doy,utHrs,glat,glon,year,altkm=altkm, CGSorSI=CGSorSI)
        Tn = outDictSI['Tn'] # in Kelvin
        MassDensity = outDictSI['MassDensity'] # kg/m^3
        AverageMass = outDictSI['AverageMass'] # kg

        # have to have a what to do if not sent in
        gz = self.GravitationalAcceleration(altkm/1000.)
        Hz = kb*Tn/(gz*AverageMass)
        return MassDensity, Hz

    def MaxwellianFlux(self, E, Q0, E0):
        """
        Input:  E energy array (eV)
                Q0 ergs/cm^2/s
                E0 eV

        Output: ergs/cm^2/s
        """
        NumFlux = Q0*E*numpy.exp(-E/E0)/2./E0**3
        EnergyFlux = E*E*NumFlux
        return NumFlux,EnergyFlux

    def MakeFangModelMatrix(self, EeV,tUnix, glat,glon,altkm = numpy.arange(80,150,1)):
        """
        This should only have to be run once to calculate what I need
        Input: unix time, longitude and latitude, altitude grid
        Do everything in SI
        """
        year = int(tUnix/(24.*3600.*365)+1970.)
        doy = int((tUnix/(24.*3600))%365)
        utHrs = (tUnix/3600.)%24
        CGSorSI = 'SI'
        outDictSI = self.msis.MSIS(doy,utHrs,glat,glon,year,altkm=altkm, CGSorSI=CGSorSI)
        Tn = outDictSI['Tn'] # in Kelvin
        MassDensity = outDictSI['MassDensity'] # kg/m^3
        AverageMass = outDictSI['AverageMass'] # kg
        # have to have a what to do if not sent in
        gz = self.GravitationalAcceleration(altkm/1000.)
        Hz = kb*Tn/(gz*AverageMass)

        # do all of the conversions here
        zcm = altkm*(1000.*100.) # cm
        EkeV = EeV/1000. # KeV
        rhoz = MassDensity*(1e3/1e6) # g/cm^3
        Hzcm = Hz*100.
        deltaE = 0.035 #keV

        # A is a matrix that is N altitude elements by M energy elements
        A = numpy.zeros([altkm.shape[0],EkeV.shape[0]-1,])

        for iEnergy in range(EkeV.shape[0]-1):
            y = (2./EkeV[iEnergy])*(rhoz*Hzcm/6e-6)**0.7 # equation 1
            Ci = self.CalculateCi(EkeV[iEnergy])
            f = Ci[0]*(y**Ci[1])*numpy.exp(-Ci[2]*(y**Ci[3]))+\
                Ci[4]*(y**Ci[5])*numpy.exp(-Ci[6]*(y**Ci[7])) # equation 4
            dE = EkeV[iEnergy+1]-EkeV[iEnergy]
            tmpqz = (f*EkeV[iEnergy]*dE)/deltaE/Hzcm
            A[:,iEnergy] = tmpqz

        # qz = f*Q0/deltaE/Hzcm # equation 3 which is wrong, look in fang 2008, eq2
        return A

    def Ionization(
            self,
            E: list[float] | numpy.typing.NDArray[float],
            EnergyFlux: list[float] | numpy.typing.NDArray[float],
            AltMin: float,
            AltMax: float,
            AltStep: float,
            tUnix: float,
            glat: float,
            glon: float,
            IonizationType='Fang'):
        """
        Parameters
        ----------
        E : list[float] | numpy.NDArray[float]
            Energy array in eV.
        EnergyFlux : list[float] | numpy.NDArray[float]
            Energy flux corresponding to each energy in E with units erg/cm^2/s or mW/m^2 (same units).
        AltMin : float
            Minimum altitude in km.
        AltMax : float
            Maximum altitude in km.
        AltStep : float
            Altitude step in km.
        tUnix : float
            Unix time.
        glat : float
            Geodetic latitude in degrees.
        glon : float
            Geodetic longitude in degrees between 0-360 degrees
        IonizationType : str, optional
            Type of ionization model to use: 'Fang', 'Maxwellian', or 'Fang+Berland'.
            The 'Fang+Berland' model uses the Fang model for energies below 10 keV and
            the Berland model for energies above 10 keV.

        Returns
        -------
        qZ : array_like
            Integrated ionization rate over energy for each altitude.
        qZE : array_like
            Ionization rate as a function of altitude and energy.
        qZsimp : array_like
            Ionization rate integrated over energy using Simpson's rule for each altitude.
            Note: I don't think that this works correctly yet.
        """
        altkm = numpy.arange(AltMin,AltMax,AltStep)
        qZE = numpy.zeros([altkm.shape[0],E.shape[0]])
        qZ = numpy.zeros(altkm.shape[0])
        qZsimp = numpy.zeros(altkm.shape[0])

        MassDensity,Hz = self.RunMSISFang(tUnix,glat,glon,altkm=altkm)

        if IonizationType == 'Fang':
            for iE in range(len(E)):
                tmpqz,y,f = self.FangModel(E[iE],EnergyFlux[iE],altkm,Hz,MassDensity)
                qZE[:,iE] = tmpqz

        elif IonizationType == 'Maxwellian':
            # this is slightly different
            # in this case would put in Q0 and E0 for each energy
            # FangModelMaxwellian(self, Q0,E0, altkm, Hz,MassDensity):
            for iE in range(len(E)):
                tmpqz,y,f    = self.FangModel(E[iE],EnergyFlux[iE],altkm,Hz,MassDensity)
                qZE[:,iE] = tmpqz

        elif IonizationType == 'Fang+Berland':
            for iE in range(len(E)):
                if E[iE] < 10 * 1e3:
                    # Fang model for below 10 keV
                    tmpqz,y,f = self.FangModel(E[iE], EnergyFlux[iE], altkm, Hz, MassDensity)
                    qZE[:,iE] = tmpqz
                else: 
                    # Berland model starts at 10 keV
                    tmpqz = self.BerlandModel(E[iE], EnergyFlux[iE], altkm, Hz, MassDensity)
                    qZE[:,iE] = tmpqz

        else:
            raise ValueError(f"Unknown IonizationType: {IonizationType}. Only 'Fang', 'Fang+Berland', or 'Maxwellian' are supported.")

        # sum over energy to get the final altitude profile
        # need to multiply by dE
        for ih in range(altkm.shape[0]):
            """
            Deprecated
            """
            qZsimp[ih] = scipy.integrate.simpson(qZE[ih,:], x=E/1000.) # new call
        # this needs to be normalized by a dimensionless quantity or I am not doing something
        # ahead of time
        qZ = numpy.sum(qZE, axis=1)

        return qZ, qZE.T,qZsimp

    def IonizationFang2008Maxwellian(self,Q0,E0,AltMin,AltMax,AltStep,tUnix,glat,glon):
        # E and EnergyFlux == same shape
        """
        Input: E0 - characteristic energy in eV
                Q0 (erg/cm^2/s or mW/m^2 - same units)
                AltMin (km)
                AltMax (km)
                AltStep (km)
                msisIn dictionary containing what need to run MSIS
        """
        altkm = numpy.arange(AltMin,AltMax,AltStep)
        MassDensity,Hz = self.RunMSISFang(tUnix,glat,glon,altkm=altkm)
        qZ,y,f = self.FangModelMaxwellian(Q0,E0,altkm,Hz,MassDensity)
        return qZ


if __name__ == '__main__':
    iz = Ionization()
    altkm = numpy.arange(50,400,1)
    Q0 = 1.0 #erg/cm^2/s
    E0 = 1*1e3 # 10 keV
    t1970 = datetime.datetime(1970,1,1,0,0,0)
    t2010 = datetime.datetime(2010,1,2,10,0,0)
    tunix = (t2010-t1970).total_seconds()
    # glat = 45
    # glon = 45
    glat = 45
    glon = 228
    MassDensity,Hz = iz.RunMSISFang(tunix,glat,glon,altkm=altkm)
    qz,y,f = iz.FangModel(E0,Q0,altkm,Hz,MassDensity)
    plt.figure(100)
    plt.semilogx(qz,altkm)
    plt.xlim([1e1,1e5])
    plt.ylim([50,400])
    plt.grid()
    plt.title('Ionization')

    Earr = numpy.logspace(2,5,31)
    plt.figure(1001)
    for iE in Earr:
        qz,y,f = iz.FangModel(iE,Q0,altkm,Hz,MassDensity)
        plt.semilogx(qz,altkm)
        plt.xlim([1e1,1e5])
        plt.ylim([50,400])
        plt.grid()

    plt.figure(101)
    plt.semilogy(f,y)
    plt.xlim([0,1])
    plt.ylim([0.1,10.])
    plt.title('f vs y')

    EeV = numpy.logspace(2,5,num=49)
    Earr = numpy.logspace(2,5,31)
    plt.figure(2001)
    for iE in Earr:
        Q0 = 1.0
        E0 = iE#5.0*1e3
        NumFlux,QeV = iz.MaxwellianFlux(EeV,Q0,E0)
        msisDict = dict()
        msisDict['tUnix'] = tunix
        msisDict['glat'] = glat
        msisDict['glon'] = glon
    # E,EnergyFlux,AltMin,AltMax,AltStep,msisIn, IonizationType='Fang'):
        qZ, qZE,qZSimp = iz.Ionization(EeV,QeV,50,400,1,tunix,glat,glon)
        plt.semilogx(qZSimp, altkm)
        plt.xlim([1e-2,1e5])
        plt.ylim([50,400])
        plt.grid()


    plt.figure(105)
    plt.plot(EeV/1000., NumFlux*(6.242e11*1000))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e1,1e7])

    altkm = numpy.arange(50,400,1)
    plt.figure(103)
    for i in range(qZE.shape[0]):
        plt.semilogx(qZE[i,:], altkm)
    plt.semilogx(qZ,altkm, 'k-', lw=2)
    plt.semilogx(qZSimp,altkm,'r-',lw=2)
    plt.xlim([1e2,1e5])
    plt.savefig('LogSpace_201points.png')
    # plt.show()

    #49 points
    NumPts = numpy.array([49,101,201,501])
    color = ['red','green','blue','purple']
    kk = 0
    plt.figure(104)
    for ii in NumPts:
        EeV = numpy.logspace(2,5,num=ii)
        Q0 = 1.0
        E0 = 5.0*1e3
        NumFlux,QeV = iz.MaxwellianFlux(EeV,Q0,E0)
        qZ, qZE,qZSimp = iz.Ionization(EeV,QeV,50,400,1,tunix,glat,glon)
        A = iz.MakeFangModelMatrix(EeV,tunix,glat,glon,altkm=altkm)
        qZMat = numpy.dot(A,NumFlux[0:-1]*6.242e11*1000.*10.) # this factor of 10 is adhoc...
        # plt.semilogx(qZ,altkm, '-', lw=2, color=color[kk])
        plt.semilogx(qZSimp,altkm,'--',lw=2,color=color[kk])
        plt.semilogx(qZMat,altkm, '+',color=color[kk] )
        plt.xlim([1e3,1e6])
        kk=kk+1

    # MakeFangModelMatrix(self, EeV,tUnix, glat,glon,altkm = numpy.arange(80,150,1)):
    EeV = numpy.logspace(2,6,num=51)
    A = iz.MakeFangModelMatrix(EeV,tunix,glat,glon,altkm=altkm)
    Q0 = 2.0
    E0 = 5.0*1e3
    NumFlux,QeV = iz.MaxwellianFlux(EeV,Q0,E0)
    dE = numpy.diff(EeV)
    plt.show()
