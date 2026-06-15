from multiprocessing import Pool
import numpy as np
from numpy import sin, cos, arcsin, arccos , tan, arctan, sqrt
from numpy import random
import matplotlib.pyplot as plt
import time as tm
import sys 
import warnings
from numba import jit
import kepler 
import bilby 
import scipy.stats as sp 
import ultranest 
from ultranest.plot import cornerplot 
import corner 
import ultranest.stepsampler 
import custom_corner_plot 

warnings.filterwarnings('ignore') 


t, x, y, ex, ey = np.loadtxt('s20-mas.txt', unpack = True) # upload position data
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

def calc_XYZTIconstX(log_a,esq,cos_i,O,w,top, m, dis): #Thiele-Innes constants 

    #n = np.sqrt(m*4.300952282/3.08567758**2/a**3)*3.155692661
    #n = (2*np.pi)/T#radians per unit time 
    m =  m*1e6
    e = np.sqrt(esq)
    a = np.exp(log_a)*206.2648
    n= sqrt(4.*np.pi*m/a**3)
    M = (n*(d-top))% (2.0*np.pi)
    #E = mikkola_solve(M,e)
    #E = kepler.solve(M,e)
    E = jit_mikkola_solve(M,e)
    #E = mikkolahybrid_solve( M,e) 
    #E= solve(M,e, 10**(-9)) 
    #E = kepler.solve(M,e)
    #E = newton_solver(M, e, tolerance=1e-9, max_iter=100, E0=None)

    E = (np.array(E))% (2.0*np.pi)
    e = np.array(e)
    eps = a*(cos(E)-e)
    eta = a*sqrt(1.-e**2)*sin(E)
    epsdot = -n*a*(sin(E)/(1.-e*cos(E)))
    etadot = n*a*sqrt(1-e**2)*(cos(E)/(1-e*cos(E)))
    i = arccos(cos_i)
    #i= np.interp(i, (0, np.pi), (0.0001, np.pi))
    A = cos(O)*cos(w) - sin(O)*sin(w)*cos(i)
    B = sin(O)*cos(w) + cos(O)*sin(w)*cos(i)
    C = sin(w)*sin(i) 
    F = -cos(O)*sin(w) - sin(O)*cos(w)*cos(i)
    G = -sin(O)*sin(w) + cos(O)*cos(w)*cos(i)
    H = cos(w)*sin(i)
    
    Y = (B*eps + G*eta)/dis #right ascension, to match with the other method and with the scale and rotate 
    X = (A*eps + F*eta)/dis #north 
    Z = (C*eps + H*eta)/dis 
    return X, Y, Z, E



def prior_transform(u):
    """Transforms the uniform random variables `u ~ Unif[0., 1.)`
    to the parameters of interest."""

    x = np.array(u)  # copy u


    x[0] = sp.uniform.ppf(u[0], loc = np.log(1), scale = np.log(200))
    x[1] = sp.uniform.ppf(u[1],loc = 0.0001,scale =0.9998 )
    x[2] = sp.uniform.ppf(u[2],loc = -1.0,scale =2.0 )
    x[3] = sp.uniform.ppf(u[3],loc = 0.0,scale = 2*np.pi )
    x[4] = sp.uniform.ppf(u[4],loc = 0.0,scale = 2*np.pi )
    x[5] = sp.uniform.ppf(u[5],loc = 2400, scale = 200 )
    x[6] = sp.norm.ppf(u[6],loc = 4.154, scale =0.014  )
    x[7] = sp.norm.ppf(u[7],loc = 8178, scale = 35 )





    return x

ndim = 8

def loglike(x):
    log_a, esq, cos_i, O, w, top, m, dis = x
    X, Y, Z, E = calc_XYZTIconstX( log_a, esq, cos_i, O, w, top, m, dis)
    loglike = -0.5 * (np.sum(((dec-X)/decerr)**2) + np.sum(((ra-Y)/raerr)**2))
    if not np.isfinite(loglike):
        loglike = -1e300
    return loglike
#best nlive 8000 nlive batch 1000, multi , slice, use stop = True 


param_names = ['log(a)', 'esq', 'cos(i)', 'Om', 'w', 'tp', 'm', 'dis'] 


sampler = ultranest.ReactiveNestedSampler(param_names, loglike, prior_transform,
                                          wrapped_params=[False, False, False, False, False, False, False,False]) 
# wrapped_params=[True, True, False, True, True, True]
#     result = sampler.run( frac_remain = 1e-5,min_num_live_points=1000,
#                      )
#     sampler.print_results()
#     sampler.plot()
#     sampler.plot_corner()
# max_iters = 1000,region_class = ultranest.mlfriends.RobustEllipsoidRegion

#min live points 2000 frac renaub 0.0001, region filter false, nsteps 20 
with Pool(4) as pool:
    sampler.stepsampler = ultranest.stepsampler.SliceSampler(
    nsteps=20,
    generate_direction=ultranest.stepsampler.generate_mixture_random_direction, region_filter = False,)
    #adaptive_nsteps='move-distance',)
    # max_nsteps=400
    result2 = sampler.run(min_num_live_points=10000,frac_remain =0.0001)
    
    flat_sampless = np.array((np.exp(result2['samples'][:,0]),np.sqrt(result2['samples'][:,1]), np.arccos(result2['samples'][:,2])*180/np.pi,result2['samples'][:,3]*180/np.pi,result2['samples'][:,4]*180/np.pi,result2['samples'][:,5],result2['samples'][:,6],result2['samples'][:,7] ))
    labels = [r"$a(mpc)$", r"$e$",  r"$i(°)$",r"$\Omega(°) $", r"$\omega(°) $", r"$t_p(yr) $",r"$m_0(10^{6}M_\odot) $",r"$D_0(pc) $"]
# print(flat_samples[ :, 0])
#    fig = corner.corner(
#    flat_sampless.T, labels=labels,quantiles=[0.16, 0.5, 0.84],
#    show_titles=True,
#    title_kwargs={"fontsize": 14},label_kwargs={"fontsize": 15},color ='blue', plot_contours = True, scale_hist = True)#,smooth1d= 2., smooth = 2.);
    
    
    fig = custom_corner_plot.custom_plot(
    flat_sampless.T, labels=labels,quantiles=None,
    show_titles=True,
    title_kwargs={"fontsize": 15},label_kwargs={"fontsize": 15},color ='blue', plot_contours = True, scale_hist = True)#,smooth1d= 2., smooth = 2.);
fig.savefig("Ultras20.pdf")    
