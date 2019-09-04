
import flopy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.sans-serif' ] = 'Calibri'
mpl.rcParams['font.size'       ] = 8
mpl.rcParams['mathtext.default'] = 'regular'          
mpl.rcParams['xtick.direction' ] = 'in'
mpl.rcParams['ytick.direction' ] = 'in'       
mpl.rcParams['lines.linewidth' ] = 0.5      

wflx = 1.
nlay, nrow, ncol, nstp = 2, 225, 140, 365
delr, delc = 90., 90.
scalar, offset = 1., 100.
hds = flopy.utils.binaryfile.HeadFile('gloc-base_0000_ADJ.hds')
adj = hds.get_alldata()
adj = np.reshape(adj[:,0,:,:], [nstp, nrow, ncol])
adj -=offset
adj /=scalar
adjsum = adj.sum(axis=0)*abs(wflx)

ibd01 = np.loadtxt('ac-cells.csv', delimiter=',')

f,s = plt.subplots(1, 1, figsize=[12.00/2.54, 14.50/2.54])
x = np.arange(0., ncol*(delc/1000.), delc/1000.)
y = np.arange(0., nrow*(delr/1000.), delr/1000.)
p = s.pcolormesh(x, y, 
                 np.ma.masked_where(ibd01==0, adjsum), 
                 cmap='viridis', edgecolor ='gray', linewidth=0.05)
s.set_ylim(s.get_ylim()[1], s.get_ylim()[0])
s.set_aspect('equal')
s.set_xlabel('Distance in x-direction (km)')
s.set_ylabel('Distance in y-direction (km)')
s.set_title('Avon model', fontsize=8)
plt.colorbar(p, ax=s)
s.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=delc))
s.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=delr))

s.xaxis.set_ticks_position('bottom')
s.yaxis.set_ticks_position('left')
for side in ['top', 'bottom', 'left', 'right']:
    s.xaxis.set_ticks_position('none')
    s.yaxis.set_ticks_position('none')
f.tight_layout()
f.savefig('Postprocess_Avon_adjoint_model_v01_211118.png', dpi=500 )
