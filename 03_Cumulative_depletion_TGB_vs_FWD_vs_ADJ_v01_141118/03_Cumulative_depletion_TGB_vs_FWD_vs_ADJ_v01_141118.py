
#%% Import packages
import flopy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import erfc, sqrt, pi, exp
from scipy.integrate import quad

#%% Define instantaneous T-G-B analytical solution (vector version)
def Theis1941GloverBalmer1954(Qw, R, t, S, T):
    D = []    
    for r in R:
        try:
            D.append(erfc(sqrt(((r**2.)*S)/(4.*T*t))))
        except:
            D.append(0.)
    return Qw*np.array(D)

#%% Define instantaneous T-G-B analytical solution (scalar version)
def Theis1941GloverBalmer1954_scalar(t, Qw, r, S, T):  
    return np.abs(Qw*erfc(sqrt(((r**2.)*S)/(4.*T*t))))

#%% Define cumulative T-G-B analytical solution (original version)
def AnderssenEtAl2017(Qw, R, t, S, T):
    D = []    
    for r in R:
        z = ((r**2.)*S)/(4.*T)
        try:
            D.append(t*erfc(sqrt(z/t))-sqrt(z/pi)*
                    (2.*((t*exp(-z/t)-sqrt(pi)*z*sqrt(t/z)*erfc(1./sqrt(t/z)))/
                    sqrt(t))))
        except:
            D.append(0.)
    return Qw*np.array(D)

#%% Define cumulative T-G-B analytical solution (alternative version)
def Cumulative_TGB(tau, Qw, b):
    return Qw*((2.*(b**2.)+tau)*erfc(b/sqrt(tau))-
              ((2.*b*sqrt(tau))/(sqrt(pi)*exp((b**2.)/tau))))

#%% Define Matplotlib parameters
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.sans-serif' ] = 'Calibri'
mpl.rcParams['font.size'       ] = 8
mpl.rcParams['mathtext.default'] = 'regular'          
mpl.rcParams['xtick.direction' ] = 'in'
mpl.rcParams['ytick.direction' ] = 'in'       
mpl.rcParams['lines.linewidth' ] = 0.5      

#%% Define plot colours
c = np.array([(120., 190.,  32.), 
              (  0.,  75., 135.), 
              (255., 184.,  28.),
              ( 65., 182., 230.),
              (228.,   0.,  43.),
              (232., 119.,  34.),
              (255., 229., 177.)])/255.

#%% Define global parameters
FWDmodelname         = '03_Cumulative_depletion_TGB_FWD'
ADJmodelname         = '03_Cumulative_depletion_TGB_ADJ'
totim                = 365.
#x_max, y_max         = 5000., 5000.
#delr, delc, top, bot = x_max/ncol, y_max/nrow, 20., 0.
nlay, nrow, ncol     = 1, 100, 100
delr, delc, top, bot = 25., 25., 50., 0. # NOTE: 50 m thick, not 20 m
thk                  = top-bot
hk, vka, ss, sy      = 1.0, np.nan, np.nan, 0.02
#hk, vka, ss, sy      = 1.0, 1.0, 0.02, 0.02
tran                 = hk*thk
Kr, Br               = hk, 1.0
Lr, Wr               = delr, delc
stg, rbt, rcol       = top-5., bot, 0
rrows                = np.arange(nrow)
cnd                  = Kr*Wr*Lr/Br
wrow, wflx           = nrow/2, -9. 
gamma, beta          = 10., 100.
#beta, gamma          = top, 1e-3 
loading              = 1.0*gamma+beta
wcols                = range(0, ncol, 10)[:5]

#%% Create output figure
f,s = plt.subplots(1,1, figsize=[16.00/2.54, 6.00/2.54])

#%% Calculate and plot numerical integration of instantaneous T-G-B solution
#x = np.arange(ncol*delr)
x = np.arange(0., ncol*delr+delr, delr)
N = []
for r in x:
    N.append(quad(Theis1941GloverBalmer1954_scalar, 0., totim, 
                  args=(wflx, r, sy, tran))[0])
s.plot(x, np.array(N)/1e3, '-', c=list(c[0]), lw=1.0,
       label='Numerical integration of instantaneous T-G-B solution')       

#%% Calculate and plot cumulative T-G-B analytical solution
#x = np.arange(ncol*delr)
x = np.arange(0., ncol*delr+delr, delr)
s.plot(x, AnderssenEtAl2017(abs(wflx), x, totim, sy, hk*thk)/1e3,
       '--', c=c[1], lw=1.0, dashes=[3,3],
       label='Closed-form cumulative T-G-B solution')

#%% Run zero extraction forward model
nper, perlen, nstp, stflag = 1, totim, int(totim), False
mf = flopy.modflow.Modflow(modelname=FWDmodelname, exe_name='../mf2005.exe')

flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper,
                         delr=delr, delc=delc, top=top, botm=bot, 
                         steady=stflag, perlen=perlen, nstp=nstp)

ibound = np.ones([nlay, nrow, ncol], dtype=int)
strt = top*np.ones([nlay, nrow, ncol], dtype=float)
flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

flopy.modflow.ModflowBcf(mf, laycon=0, tran=tran, sf1=sy) 

spd = {}
lst = []
for rrow in rrows:
    lst.append([0, rrow, rcol, stg, cnd, rbt])
spd.update({0: lst})
flopy.modflow.ModflowRiv(mf, ipakcb=1, stress_period_data=spd) 

flopy.modflow.ModflowPcg(mf, hclose=1e-3, rclose=1e-3)

spd = {}
for ts in range(nstp):
    spd.update( {(0, ts): ['print head', 'save head', 'save budget']})
flopy.modflow.ModflowOc(mf, stress_period_data=spd)
    
mf.write_input()
success, buff = mf.run_model(silent=True)

flx = flopy.utils.binaryfile.CellBudgetFile(FWDmodelname+'.cbc')
flx = flx.get_data(text='   RIVER LEAKAGE', full3D=True)
flx = np.array(flx)[:,0,rrows,rcol]
QRwp1 = flx.sum()

#%% Run forward models for each bore-stream separation distance
nper, perlen, nstp, stflag = 1, totim, int(totim), False
QRwp2 = []
depl_FWD = []
wcs = range(ncol) 
x = np.arange(0, ncol)*delc
xw = x[wcols]
for i,wcol in enumerate(wcols):
    mf = flopy.modflow.Modflow(modelname=FWDmodelname, 
                               exe_name='../mf2005.exe')
    
    flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper, 
                             delr=delr, delc=delc, top=top, botm=bot, 
                             steady=stflag, perlen=perlen, nstp=nstp)
    
    ibound = np.ones([nlay, nrow, ncol], dtype=int)
    strt = top*np.ones([nlay, nrow, ncol], dtype=float)
    flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    
    flopy.modflow.ModflowBcf(mf, laycon=0, tran=tran, sf1=sy) 
    
    spd = {}
    lst = []
    for rrow in rrows:
        lst.append([0, rrow, rcol, stg, cnd, rbt])
    spd.update({0: lst})
    flopy.modflow.ModflowRiv(mf, ipakcb=1, stress_period_data=spd) 
    
    flopy.modflow.ModflowPcg(mf, hclose=1e-3, rclose=1e-3)
    
    spd = {}
    for ts in range(nstp):
        spd.update( {(0, ts): ['print head', 'save head', 'save budget']})
    flopy.modflow.ModflowOc(mf, stress_period_data=spd)
    
    spd = {0: [[0, wrow, wcol, wflx]]}
    flopy.modflow.ModflowWel(mf, stress_period_data=spd)
    
    mf.write_input()
    success, buff = mf.run_model(silent=True)
    
    flx = flopy.utils.binaryfile.CellBudgetFile(FWDmodelname+'.cbc')
    flx = flx.get_data(text='   RIVER LEAKAGE', full3D=True)
    flx = np.array(flx)[:,0,rrows,rcol]
    QRwp2.append(flx.sum())
    depl_FWD.append(abs(QRwp1-QRwp2[-1])/1000.)
    if i==0:
        lab='Numerical forward model solutions'
    else:
        lab='_no_label'
    s.plot(xw[i], depl_FWD[-1], 'o', ms=5, mfc=list(c[2]), mec='none', 
           label=lab, zorder=0)       

QRwp2 = np.array(QRwp2)
depl_FWD = np.array(depl_FWD)

#%% Run single adjoint model
nstp_scalar = 1
mf = flopy.modflow.Modflow(modelname=ADJmodelname, exe_name='../mf2005.exe')

nper, perlen, nstp, stflag = 1, totim, int(totim), False

flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper,
                         delr=delr, delc=delc, top=top, botm=bot, 
                         steady=stflag, perlen=perlen, nstp=nstp*nstp_scalar)
                                
ibound = np.ones([nlay, nrow, ncol], dtype=int)
#ibound[0,  0,  :] = -1
#ibound[0, -1,  :] = -1
ibound[0,  :, -1] = -1
strt = beta*np.ones([nlay, nrow, ncol], dtype=float)
strt[0, rrows, rcol] = cnd/delr/delc/sy*gamma+beta
flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

#lst = []
#for row in rrows:
#    lst.append([0, row, rcol, cnd/delr/delc/sy*gamma+beta, 
#                              cnd/delr/delc/sy*gamma+beta])
#spd = {0: lst}
#chd = flopy.modflow.ModflowChd(mf, stress_period_data=spd)

flopy.modflow.ModflowBcf(mf, laycon=0, tran=tran, sf1=sy) 

lst = []
for row in rrows:
    lst.append([0, row, rcol, loading, cnd, rbt])
#    lst.append([0, row, rcol, cnd/delr/delc/sy*gamma+beta, cnd, rbt])
spd = {0: lst}
flopy.modflow.ModflowRiv(mf, stress_period_data=spd) 

flopy.modflow.ModflowPcg(mf, hclose=1e-3, rclose=1e-3)

spd = {}
for ts in range(nstp):
    spd.update( {(0, ts): ['print head', 'save head', 'print budget', 
                           'save budget']})
flopy.modflow.ModflowOc(mf, stress_period_data=spd)
      
mf.write_input()
success, buff = mf.run_model(silent=True)

hds = flopy.utils.binaryfile.HeadFile(ADJmodelname+'.hds')
adj = hds.get_alldata()
adj = np.reshape(adj, [nstp, nrow, ncol])
adj = adj-beta
adj = adj/gamma/nstp_scalar
adjsum = adj.sum(axis=0)*abs(wflx)/1000.
hds.close()

x = np.arange(0, ncol)*delc
s.plot(x[::2], adjsum[wrow,:][::2], 'o', ms=5, mec=list(c[1]), mfc='none', 
       mew=0.5, label='Numerical adjoint model solution')

#%% Apply plot formatting and finish
s.legend(numpoints=1, ncol=1, labelspacing=0.75, fancybox=False)
s.set_xlim(-50, 2050)
s.set_xticklabels(s.get_xticks()/1000.)
s.set_xlabel('Bore$-$stream separation distance (km)')
s.set_ylabel('Cumulative streamflow depletion (ML)')
s.xaxis.set_ticks_position('bottom')
s.yaxis.set_ticks_position('left')

for side in ['top', 'left', 'right']:
    s.spines[side].set_visible(False)
    s.yaxis.set_ticks_position('none')
    s.grid(which='major', axis='y', 
           c=(194./255., 194./255., 194./255.), ls='-', lw=0.5)

f.tight_layout()
f.savefig('03_Cumulative_depletion_TGB_vs_FWD_vs_ADJ_v01_141118.png', dpi=500)
plt.close(f)

#%%
# Remove Modflow input and output files
from os import getcwd, listdir, remove
for f in listdir(getcwd()):
    if f[-3:] in ['bas', 'bcf', 'cbc', 'dis', 'hds', 
                  'ist', 'nam', '.oc', 'pcg', 'riv', 'wel']:
        remove(f)
