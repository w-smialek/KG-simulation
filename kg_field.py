import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from time import time
import py_solver_field as solver

N2 = 75            # max positive/negative mode in px and py
Ntot = 2*N2         # Total number of modes in any dimension

###
### Relations between program variables and physical variables
###

m = 1.0             # mass in multiples of m_e
L = 100             # torus half-diameter in x and y in multiples of hbar/(m_e * c)
                    # (that means THE TOTAL LENGTH IS 2L)

# time_phys = time_var * hbar/(c^2 * m_e)
# p_phys    = p_var    * m_e * c
# ene_phys  = ene_var  * m_e * c^2
# x_phys    = x_var    * hbar/(m_e * c)

###
### Prepare initial conditions
###

# Parameters

pot0=530 # electric potential multiplicator
x0=40     # position of the x-center of wavepacket
y0=0     # position of the y-center of wavepacket
r0=40      # optionally - ring-shaped initial field (commented parts of the code)
vx0=0.0    # optionally - position of the x-center of potential field (commented parts of the code)
vy0=0.0    # optionally - position of the y-center of potential field (commented parts of the code)
px0=0.0   # mean x-momentum of the wavepacket
py0=0  # mean y-momentum of the wavepacket

sim = solver.kgsim(L,m,N2)

print('Length: %.1f hbar/mc'%L)
print('Momentum range: +-%.3f mc'%sim.p_extent_hi)

# Initial KG field

a_gauss = 700
phi_bar = np.zeros((2,Ntot,Ntot)).astype(complex)

phi_bar[0,...] = (np.exp(-1j*(x0*sim.space_px + y0*sim.space_py) \
                          -a_gauss/sim.p_extent_hi*((sim.space_px - px0)**2 + (sim.space_py - py0)**2)))[0,...]
phi_bar[1,...] = (np.exp(-1j*(x0*sim.space_px + y0*sim.space_py) \
                          -a_gauss/sim.p_extent_hi*((sim.space_px + px0)**2 + (sim.space_py + py0)**2)))[0,...]

plt.imshow(phi_bar[0,...].real)
plt.show()

# phi_bar[1,...] = np.exp(-1j*(x0*sim.space_px + y0*sim.space_py) - 10*(sim.space_px**2 + sim.space_py**2))[0,...]
# phi_bar[1,...] = 0
# phi_bar[1,N2+1,N2] = 1

# OPTIONALLY - Transform between field representations.
# Solver requires the Feshbach-Villard representation

# idtvarphi = np.exp(-0.01*( np.sqrt(sim.space_x**2 + sim.space_y**2)[0,...] - r0 )**2)
# varphi = np.exp(-0.01*( np.sqrt(sim.space_x**2 + sim.space_y**2)[0,...] - r0 )**2)

# psi = np.zeros((2,Ntot,Ntot)).astype(complex)
# psi[0,...] = 1/np.sqrt(2)*(varphi+idtvarphi)
# psi[1,...] = 1/np.sqrt(2)*(varphi-idtvarphi)

# phi = sim.psi_to_phi(psi)
# phi_bar = sim.phi_to_phi_bar(phi)    

# Normalize charges

qp,qn,qt = sim.charges(phi_bar)

# set magnitude of the initial positive- and negative charge

relative_qneg = 0
relative_qpos = 1

phi_bar[0,...] *= np.sqrt(abs(relative_qpos/qp))
phi_bar[1,...] *= np.sqrt(abs(relative_qneg/qn))

plt.imshow(sim.psi_to_rho(*sim.phi_to_psi(sim.phi_bar_to_phi(phi_bar))))
plt.show()

qp,qn,qt = sim.charges(phi_bar,printit=True)

print("Wavepacket mean momentum: %.3f mc"%(2*np.sqrt(px0**2 + py0**2)))

# Potential

a_gauss = 0.0003

# pypotential = -pot0/(np.sqrt(sim.space_x[0,...]**2 + sim.space_y[0,...]**2)+20)
# pypotential[:,:] = 0

#

# pypotential = 0.3*(np.exp(- a_gauss*10/L*((sim.space_x[0,...] - x_0)**2 + (sim.space_y[0,...] - y_0)**2)))
# pypotential += 0.3*(np.exp(- a_gauss*10/L*((sim.space_x[0,...] + x_0)**2 + (sim.space_y[0,...] + y_0)**2)))
# pypotential = pot0/10*(np.exp(- a_gauss*((sim.space_x[0,...] - vx0)**2 + (sim.space_y[0,...] - vy0)**2)))
pypotential = -pot0*1/137/np.sqrt((sim.space_x[0,...])**2 + (sim.space_y[0,...])**2)
regval = pypotential[N2,N2+1]
pypotential[N2,N2] = regval
#

# wd = 10

# pypotential = np.zeros((Ntot,Ntot))
# pypotential[:,3*N2//4+wd:5*N2//4-wd] = pot0
# pypotential[:,3*N2//4:3*N2//4+wd] = np.fromfunction(lambda i, j: j/wd*pot0, (Ntot, wd))
# pypotential[:,5*N2//4-wd:5*N2//4] = np.fromfunction(lambda i, j: (1-j/wd)*pot0, (Ntot, wd))

#

print("Max. abs potential: %.3f mc^2/e"%np.max(abs(pypotential)))

plt.imshow(pypotential, origin='lower', extent=sim.x_extent)
plt.show()

###
### Run solver
###

blocks = 20
total_time = 1200
total_timesteps = 600

t1 = time()
for iter in range(blocks):

    print("=== Block %i, current time: %i ===" %(iter, total_time//blocks * iter))

    sim = solver.kgsim(L,m,N2)

    t_span = (0., total_time/blocks)
    n_timesteps = total_timesteps//blocks

    t0 = time()
    sim.solve(pypotential, phi_bar,t_span,n_timesteps,kgform='phibar')
    te = time()
    print("Mycyrk time: %f"%(te-t0))

    print(sim.result.message)
    print("Size of solution: ", sim.result.size)

    t0 = time()
    phi_bar = sim.save('./solutions/sol%i.npy'%iter,destroy_cyrk=True)
    te = time()
    print("Saving time: %f"%(te-t0))
t2 = time()
print('TOTAL SOLVE TIME: %f'%(t2-t1))

###
### Render pictures
###

t0 = time()
sim = solver.kgsim(L,m,N2)
for iter in range(blocks):
    sim.load('./solutions/sol%i.npy'%iter)
te = time()
print("Loading time: %f"%(te-t0))


factor = 2                  # factor for interpolation grid density
pb_complex_plot = False     # complex plot of the Feshbach-Villard representation
vp_complex_plot = False     # complex plot of the Klein-Gordon field
vp_abs_plot = False         # absolute value plot of the complex Klein-Gordon field
charge_plot = True          # charge density plot
fps = 30                    # gif frames per second

gif_id = "_valid_scatter_pot%.2f_mom%.2f_"%(pot0,np.sqrt((px0)**2+(py0)**2))
cmap_str = 'seismic'
charge_satur_val = 2*max(abs(qp),abs(qn))*5/L**2   # value at which colormap of the charge plot should saturate,
                                                    # 0 -> automatic

load_tsteps = np.linspace(0,total_time,total_timesteps)

t0 = time()
sim.render(factor,fps,gif_id,pb_complex_plot,vp_complex_plot,
           vp_abs_plot,charge_plot,cmap_str,charge_satur_val,
           fromloaded=load_tsteps)
te = time()
print("rendering images time: %f"%(te-t0))
