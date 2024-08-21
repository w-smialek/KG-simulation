import numpy as np
import matplotlib.pyplot as plt
from time import time
import py_solver_field as solver

N2 = 100            # max positive/negative mode in px and py
Ntot = 2*N2         # Total number of modes in any dimension

###
### Relations between program variables and physical variables
###

m = 1.0             # mass in multiples of m_e
L = 400             # torus half-diameter in x and y in multiples of hbar/(m_e * c)
                    # (that means THE TOTAL LENGTH IS 2L)

# time_phys = time_var * hbar/(c^2 * m_e)
# p_phys    = p_var    * m_e * c
# ene_phys  = ene_var  * m_e * c^2
# x_phys    = x_var    * hbar/(m_e * c)

###
### Prepare initial conditions
###

# Parameters

pot0=-10.0
x0=0.0
y0=0.0
r0=40
vx0=0.0
vy0=0.0
px0=0.0
py0=0.0
px0=px0/2 #    SO THIS IS ACTUALLY HALF MOMENTUM, 
py0=py0/2 #    NEED TO ANALYZE HOW THE UNITS AND FACTORS OF 2 PLAYS EXACTLY

sim = solver.kgsim(L,m,N2)

print('Length: %.1f hbar/mc'%L)
print('p_extent: %.3f mc'%sim.p_extent_hi)

# Initial KG field

a_gauss = 50
phi_bar = np.zeros((2,Ntot,Ntot)).astype(complex)

# phi_bar[0,...] = (np.exp(-1j*(x0*sim.space_px + y0*sim.space_py) -a_gauss/sim.p_extent_hi*((sim.space_px - px0)**2 + (sim.space_py - py0)**2)))[0,...]
# phi_bar[1,...] = (np.exp(-1j*(x0*sim.space_px + y0*sim.space_py) -a_gauss/sim.p_extent_hi*((sim.space_px - px0)**2 + (sim.space_py - py0)**2)))[0,...]
# phi_bar[1,...] = (np.exp(-1j*(x0*sim.space_px + y0*sim.space_py) - a_gauss/sim.p_extent_hi*((sim.space_px + px0)**2 + (sim.space_py + py0)**2)))[0,...]

# Transform between field representations.
# Solver requires the Feshbach-Villard representation

idtvarphi = np.exp(-0.01*( np.sqrt(sim.space_x**2 + sim.space_y**2)[0,...] - r0 )**2)
varphi = np.exp(-0.01*( np.sqrt(sim.space_x**2 + sim.space_y**2)[0,...] - r0 )**2)

psi = np.zeros((2,Ntot,Ntot)).astype(complex)
psi[0,...] = 1/np.sqrt(2)*(varphi+idtvarphi)
psi[1,...] = 1/np.sqrt(2)*(varphi-idtvarphi)

phi = sim.psi_to_phi(psi)
phi_bar = sim.phi_to_phi_bar(phi)    

# Normalize charges

qp,qn,qt = sim.charges(phi_bar)
phi_bar[0,...] *= np.sqrt(abs(0/qp))
phi_bar[1,...] *= np.sqrt(abs(1/qn))

plt.imshow(sim.psi_to_rho(*sim.phi_to_psi(sim.phi_bar_to_phi(phi_bar))))
plt.show()

qp,qn,qt = sim.charges(phi_bar,printit=True)

print("Wavepacket mean momentum: %.3f mc"%(2*np.sqrt(px0**2 + py0**2)))

# Potential

a_gauss = 0.0003

# pypotential = 0.3*(np.exp(- a_gauss*10/L*((sim.space_x[0,...] - x_0)**2 + (sim.space_y[0,...] - y_0)**2)))
# pypotential += 0.3*(np.exp(- a_gauss*10/L*((sim.space_x[0,...] + x_0)**2 + (sim.space_y[0,...] + y_0)**2)))
# pypotential = pot0/10*(np.exp(- a_gauss*((sim.space_x[0,...] - vx0)**2 + (sim.space_y[0,...] - vy0)**2)))

pypotential = -pot0/(np.sqrt(sim.space_x[0,...]**2 + sim.space_y[0,...]**2)+20)

# pypotential = np.zeros((Ntot,Ntot))
# pypotential[:,N2//2+25:3*N2//2-25] = pot0

print("Max. abs potential: %.3f mc^2/e"%np.max(abs(pypotential)))

plt.imshow(pypotential, origin='lower', extent=sim.x_extent)
plt.show()

###
### Run solver
###

blocks = 10
total_time = 1000
total_timesteps = 500

for iter in range(blocks):

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
    print(iter)

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
stretch = 1                 # factor for resizing final image, no interpolation
pb_complex_plot = False     # complex plot of the Feshbach-Villard representation
vp_complex_plot = False     # complex plot of the Klein-Gordon field
vp_abs_plot = False         # absolute value plot of the complex Klein-Gordon field
charge_plot = True          # charge density plot
fps = 20                    # gif frames per second

gif_id = "_hydro_pot%.2f_mom%.2f_"%(pot0,px0)
cmap_str = 'seismic'
charge_satur_val = 30*max(abs(qp),abs(qn))*5/L**2   # value at which colormap of the charge plot should saturate,
                                                    # 0 -> automatic

load_tsteps = np.linspace(0,total_time,total_timesteps)

t0 = time()
sim.render(factor,stretch,fps,gif_id,pb_complex_plot,vp_complex_plot,
           vp_abs_plot,charge_plot,cmap_str,charge_satur_val,
           fromloaded=load_tsteps)
te = time()
print("rendering images time: %f"%(te-t0))
