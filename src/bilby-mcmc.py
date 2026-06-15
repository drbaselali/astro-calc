from multiprocessing import Pool
import numpy as np
from numpy import sin, cos, arcsin, arccos , tan, arctan, sqrt
from numpy import random
import matplotlib.pyplot as plt
import time as tm
import corner 
import sys 
import warnings
from numba import jit
import kepler 
import bilby 

import custom_corner_plot 
warnings.filterwarnings('ignore') 



t, x, y, ex, ey = np.loadtxt('testlong.txt', unpack = True) # upload position data
data = np.column_stack((t, x, y, ex, ey))#in marcsec
d = data[:,0]
d2 = 2002.5
ra = data[:,1]/1000#mpc 
raerr = data[:,3]/1000
dec = data[:,2]/1000
decerr = data[:,4]/1000
#print(ra)

# calculating proper motion and saving the result to a file
rav = []
decv = []
raverr = []
decverr = [] 
for i in range(len(ra[:-1])):
    ravel = (ra[i+1] - ra[i])/(d[i+1]-d[i])
    decvel = (dec[i+1] - dec[i])/(d[i+1] - d[i])
    raver =  (raerr[i+1])+ (raerr[i])#/(d[i+1]-d[i]) 
    decver = (decerr[i+1])+ (decerr[i])#/(d[i+1]-d[i])  
    rav = np.append(rav,(ravel))
    decv = np.append(decv,(decvel))
    raverr = np.append(raverr,(raver))
    decverr = np.append(decverr,(decver))
    #np.savetxt('propermotion.txt',velowitherr)
rav = np.append(rav,(np.mean(ravel)))  
decv = np.append(decv,(np.mean(decvel)))    
raverr = np.append(raverr,(np.mean(raver)))
decverr = np.append(decverr,(np.mean(decver)))
@jit
def jit_mikkola_solve(M,e):##mikkola solver
    alpha = (1.0 - e) / ((4.*e) + 0.5)#constants according to the paper 
    beta = (0.5*M) / ((4.*e) + 0.5)
    ab = np.sqrt(beta**2. + alpha**3.)
    z = np.abs(beta + ab)**(1./3.)

    s1 = z - alpha/z
    ds = -(0.078 * (s1**5)) / (1 + e)
    s = s1 + ds

    E0 = M + e * ( 3.*s - 4.*(s**3.) ) # cubic form of Kepler's equation 
    sinE = np.sin(E0)
    cosE = np.cos(E0)

    f = E0 - e*sinE - M
    fp = 1. - e*cosE #derivatives 
    fpp = e*sinE
    fppp = e*cosE
    fpppp = -fpp

    co1 = -f / fp #correction terms 
    co2 = -f / (fp + 0.5*fpp*co1)
    co3 = -f / ( fp + 0.5*fpp*co2 + (1./6.)*fppp*(co2**2) )
    co4 = -f / ( fp + 0.5*fpp*co3 + (1./6.)*fppp*(co3**2) + (1./24.)*(fpppp)*(co3**3) )

    return E0 + co4

def solve( M0, e, h):#h is the tolernce level, this one is used in Marzieh's code 
    if e < 0.8: #
        E0 = M0#good for small e since E differs from M by a term of order e 
    else:
        E0 = np.pi #to improve convergence
        
   
    for i in range(50):
        E = E0-((E0 - e*np.sin(E0) - M0)/(1 -e*np.cos(E0)))
        if abs(np.any(E - E0)) < h:
            break
        else:
            E0 = E
    return E
#@jit
def newton_solver(M, e, tolerance=1e-9, max_iter=10000, E0=None):

    M = np.asarray(M)
    e = np.asarray(e)

    if E0 is None:
        E = np.copy(M)
    else:
        E = np.copy(E0)


    E -= (E - (e * np.sin(E)) - M) / (1.0 - (e * np.cos(E)))

    diff = (E - (e * np.sin(E)) - M) / (1.0 - (e * np.cos(E)))
    abs_diff = np.abs(diff)
    ind = np.where(abs_diff > tolerance)
    niter = 0
    while ((ind[0].size > 0) and (niter <= max_iter)):
        E[ind] -= diff[ind]
        if niter == (max_iter//2):
            E[ind] = np.pi
        diff[ind] = (E[ind] - (e[ind] * np.sin(E[ind])) - M[ind]) / \
            (1.0 - (e[ind] * np.cos(E[ind])))
        abs_diff[ind] = np.abs(diff[ind])
        ind = np.where(abs_diff > tolerance)
        niter += 1

    if niter >= max_iter:
        E[ind] = jit_mikkola_solve(M[ind], e[ind]) 
    return E

def calc_XYZTIconst(log_a, esq, cos_i, O, w, tp, date, m , dis): #Thiele-Innes constants 
    #m = 4.154e6
    #m_sgra_err = 0.014e6 
    #n = np.sqrt(m*4.300952282/3.08567758**2/a**3)*3.155692661
    #T = np.sqrt((np.absolute(a)**3)/np.absolute(m))
    #n = (2*np.pi)/T#radians per unit time
    e = np.sqrt(esq)
    m = m*1e6
    a = np.exp(log_a)*206.2648
    i = arccos(cos_i)
    n= sqrt(4.*np.pi*m/a**3)
    M = (n*(date-tp))% (2.0*np.pi)
    

    #E = mikkola_solve(M,e)
    #E = kepler.solve(M,e)
    E = jit_mikkola_solve(M,e)
    #E = mikkolahybrid_solve( M,e) 
    #E= solve(M,e, 10**(-9)) 
    #E = kepler.solve(M,e)
    #E = newton_solver(M, e, tolerance=1e-9, max_iter=100, E0=None)

    E = (np.array(E))% (2.0*np.pi)
    e = np.array(e)
    eps = a*(np.cos(E)-e)
    eta = a*sqrt(1.-e**2)*np.sin(E)
    epsdot = -n*a*(np.sin(E)/(1.-e*np.cos(E)))
    etadot = n*a*sqrt(1-e**2)*(np.cos(E)/(1-e*np.cos(E)))
    A = np.cos(O)*np.cos(w) - np.sin(O)*np.sin(w)*np.cos(i)
    B = np.sin(O)*np.cos(w) + np.cos(O)*np.sin(w)*np.cos(i)
    C = np.sin(w)*np.sin(i) 
    F = -np.cos(O)*np.sin(w) - np.sin(O)*np.cos(w)*np.cos(i)
    G = -np.sin(O)*np.sin(w) + np.cos(O)*np.cos(w)*np.cos(i)
    H = np.cos(w)*np.sin(i)
    
    Y = (B*eps + G*eta)/dis #right ascension
    X = (A*eps + F*eta)/dis #north 
    Z = (C*eps + H*eta)/dis 
    
    return X, Y, Z, E


def calc_XYZ(log_a, e, i, O, w, tp, date): #needs fixing like the above if used 

    m = 4.154e6
    #m_sgra_err = 0.014e6 
    #n = np.sqrt(m*4.300952282/3.08567758**2/a**3)*3.155692661
    #T = np.sqrt((np.absolute(a)**3)/np.absolute(m))
    #n = (2*np.pi)/T#radians per unit time
    a = np.exp(log_a)
    n= sqrt(4.*np.pi*m/a**3)
    M = (n*(date-tp))% (2.0*np.pi)
    

    #E = mikkola_solve(M,e)
    E = kepler.solve(M,e)
    #E = jit_mikkola_solve(M,e)
    #E = mikkolahybrid_solve( M,e) 
    #E= solve(M,e, 10**(-9)) 
    #E = kepler.solve(M,e)
    #E = newton_solver(M, e, tolerance=1e-9, max_iter=100, E0=None)

    E = (np.array(E))% (2.0*np.pi)
    e = np.array(e)
    
    #f1 = sqrt(1.+e)*sin(E/2.)
    #f2 = sqrt(1.-e)*cos(E/2.)
    #f = (np.degrees(2.*np.arctan2(f1,f2)))%360
    bet = e/(1.+sqrt(1.-e**2))
    f = E + 2. * arctan(bet*sin(E)/(1.-bet*cos(E)))
    #f = np.radians(f)
    #r = a*(1-e*cos(E))
    r = (a*(1.-e**2))/(1.+(e*cos(f)))#cite Murray & Correia (2010) equations 53,54, and 55
    X = r * ( cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i) ) #north, to match with the 3d plots, in which X is north 
    Y = r * ( sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i) ) #+ra, -ra = -y
    Z = r * (sin(w+f)*sin(i))# +z towards the observer 
    #plt.hist(cos(E))
    return X, Y, Z, E


priors  = dict()

priors ["log_a"] = bilby.core.prior.Uniform(np.log(1), np.log(100), "log_a")
priors ["esq"] = bilby.core.prior.Uniform(0.0001, 0.9998, "esq")
priors ["cos_i"] = bilby.core.prior.Uniform(-1, +1, "cos_i") 
priors ["O"] = bilby.core.prior.Uniform(0, 2*np.pi, "O") 
priors ["w"] = bilby.core.prior.Uniform(0, 2*np.pi, "w") 
priors ["tp"] = bilby.core.prior.Uniform(2000, 2050, "tp") 
priors ["m"] = bilby.core.prior.Gaussian(mu = 4.154 , sigma = 0.014 ,name ="m") 
priors ["dis"] = bilby.core.prior.Gaussian(mu = 8178 , sigma = 35, name = "dis") 


class chiLikelihood(bilby.Likelihood):
    def __init__(self, d, dec, ra, decerr, raerr):
        super().__init__(parameters={'log_a': None, 'esq': None,'cos_i': None, 'O': None,'w': None, 'tp': None,'m': None,'dis': None})
        self.d = d
        self.dec = dec 
        self.ra = ra 
        self.decerr = decerr 
        self.raerr = raerr
        self.N = len(d)

    def log_likelihood(self):
        log_a = self.parameters['log_a']
        esq = self.parameters['esq']
        cos_i = self.parameters['cos_i']
        O = self.parameters['O']
        w = self.parameters['w']
        tp = self.parameters['tp']
        m = self.parameters['m']
        dis = self.parameters['dis']
        X, Y, Z, E = calc_XYZTIconst( log_a, esq, cos_i, O, w, tp, d,m,dis)


        return -0.5 * (np.sum(((dec-X)/decerr)**2) + np.sum(((ra-Y)/raerr)**2))


    
#  




print(priors)
likelihood =  chiLikelihood(d, dec, ra, decerr, raerr)

# Tmax 10000, min tau 25, t0 1000, nu 1000, l1 150, l2 50 best for 500 samples  

#L1steps=1000, L2steps=100,,pt_ensemble = True,
#        pt_rejection_sample = False, adapt = True,Tmax = 10000, min_tau = 20,
#     adapt_t0 = 1000, adapt_nu = 1000,

# maximize L1steps 50 best
result = bilby.run_sampler(nsamples=900,
        stop_after_convergence=True,
        ntemps=15,npool=4,
        printdt=50,check_point_delta_t=5000,
    resume=False,
    likelihood=likelihood,
    priors=priors,
    sampler="bilby_mcmc"
   
)


flat_sampless = np.array((np.exp(result.samples['log_a']),np.sqrt(result.samples['esq']), np.arccos(result.samples['cos_i'])*180/np.pi,result.samples['O']*180/np.pi,result.samples['w']*180/np.pi,result.samples['tp'],result.samples['m'],result.samples['dis'] ))
labels = [r"$a(mpc)$", r"$e$",  r"$i(°)$",r"$\Omega(°) $", r"$\omega(°) $", r"$t_p(yr) $",r"$m_0(10^{6}M_\odot) $",r"$D_0(pc) $"]
# print(flat_samples[ :, 0])
#fig = corner.corner(
#    flat_sampless.T, labels=labels,quantiles=[0.16, 0.5, 0.84],
#    show_titles=True,
#    title_kwargs={"fontsize": 14},label_kwargs={"fontsize": 15},color='blue', plot_contours= True, scale_hist = True)#,smooth1d= 2., smooth = 2.);


fig = custom_corner_plot.custom_plot(
    flat_sampless.T, labels=labels,quantiles=None,
    show_titles=True,
    title_kwargs={"fontsize": 15},label_kwargs={"fontsize": 15},color ='gray', plot_contours = True, scale_hist = True)#,smooth1d= 2., smooth = 2.);
    

fig.savefig("bilbylong.pdf")




 

