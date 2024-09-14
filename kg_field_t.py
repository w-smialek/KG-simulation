
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from time import time
import py_solver_field as solver
import scipy.integrate
import scipy.interpolate

N2 = 40            # max positive/negative mode in px and py
Ntot = 2*N2         # Total number of modes in any dimension

###
### Relations between program variables and physical variables
###

m = 1.0             # mass in multiples of m_e
L = 200             # torus half-diameter in x and y in multiples of hbar/(m_e * c)
                    # (that means THE TOTAL LENGTH IS 2L)

# time_phys = time_var * hbar/(c^2 * m_e)
# p_phys    = p_var    * m_e * c
# ene_phys  = ene_var  * m_e * c^2
# x_phys    = x_var    * hbar/(m_e * c)

###
### Prepare initial conditions
###

# Parameters

pot0=-3 # electric potential multiplicator
x0=50     # position of the x-center of wavepacket
y0=0     # position of the y-center of wavepacket
r0=40      # optionally - ring-shaped initial field (commented parts of the code)
vx0=0.0    # optionally - position of the x-center of potential field (commented parts of the code)
vy0=0.0    # optionally - position of the y-center of potential field (commented parts of the code)
px0=0.0   # mean x-momentum of the wavepacket
py0=0.4  # mean y-momentum of the wavepacket

sim = solver.kgsim(L,m,N2)

print('Length: %.1f hbar/mc'%L)
print('Momentum range: +-%.3f mc'%sim.p_extent_hi)

# Initial KG field

a_gauss = 70
phi_bar = np.zeros((2,Ntot,Ntot)).astype(complex)

phi_bar[0,...] = (np.exp(-1j*(x0*sim.space_px + y0*sim.space_py) \
                          -a_gauss/sim.p_extent_hi*((sim.space_px - px0)**2 + (sim.space_py - py0)**2)))[0,...]
phi_bar[1,...] = (np.exp(-1j*(x0*sim.space_px + y0*sim.space_py) \
                          -a_gauss/sim.p_extent_hi*((sim.space_px + px0)**2 + (sim.space_py + py0)**2)))[0,...]

plt.imshow(phi_bar[0,...].real)
plt.show()

# Normalize charges

qp,qn,qt = sim.charges(phi_bar)

relative_qneg = 0
relative_qpos = 1

phi_bar[0,...] *= np.sqrt(abs(relative_qpos/qp))
phi_bar[1,...] *= np.sqrt(abs(relative_qneg/qn))

plt.imshow(sim.psi_to_rho(*sim.phi_to_psi(sim.phi_bar_to_phi(phi_bar))))
plt.show()

qp,qn,qt = sim.charges(phi_bar,printit=True)

print("Wavepacket mean momentum: %.3f mc"%(2*np.sqrt(px0**2 + py0**2)))

# Potential

n_t = 200
total_time = 800
t_linspace = np.linspace(0,total_time,n_t)#-400:0.2:400
E0 = -0.1
T_0 = 40
sigma = 5
M = 1
E_ar = E0*(np.exp(-((t_linspace-total_time/2-(T_0/2))/sigma)**(2*M)) - np.exp(-((t_linspace-total_time/2+(T_0/2))/sigma)**(2*M)))
A_ar = np.flip(scipy.integrate.cumulative_trapezoid(np.flip(E_ar),t_linspace,initial=0))

# pypotential_a1_interp = scipy.interpolate.CubicSpline(t_linspace,pypotential_a1,axis=2).c

# t_interp = np.linspace(-400,400,n_t*3)#-400:0.2:400
# nar = np.zeros(len(t_interp))

# for i_t in range(len(t_interp)):
#     for i_k in range(len(t_linspace)):
#         if t_linspace[i_k] <= t_interp[i_t] < t_linspace[i_k +1] or i_k == n_t-2:
#             nar[i_t] = sum([pypotential_a1_interp[p,i_k,45,20]*(t_interp[i_t]-t_linspace[i_k])**(3-p) for p in range(4)])
#             break

# plt.plot(t_linspace,A_ar)
# plt.plot(t_interp, nar)
# plt.show()

X, Y, T = np.meshgrid(sim.x_linspace,sim.x_linspace,t_linspace)

pypotential = -pot0*np.exp(-0.01*(X**2 + Y**2))*0

pypotential_a1 = -1/2*pot0*np.sin(Y/L*np.pi)*np.sin(T/total_time*2*np.pi)*0

pypotential_a2 = 1/2*pot0*np.sin(X/L*np.pi)*np.sin(T/total_time*2*np.pi)*0

pypotential_a1 = np.tile(A_ar,(Ntot,Ntot,1))

pypotential_a0 = pypotential_a1**2 + pypotential_a2**2

t_int = np.linspace(0,total_time,10*n_t)
curv = pypotential_a1[N2+20,N2+20,:]
splines = scipy.interpolate.CubicSpline(t_linspace,curv)
curv_i = splines(t_int)
plt.plot(t_linspace,curv)
plt.plot(t_int,curv_i)
plt.show()

print("Max. abs potential: %.3f mc^2/e"%np.max(abs(pypotential)))

# plt.imshow(pypotential[...,n_t//2], origin='lower', extent=sim.x_extent)
# plt.show()

# plt.imshow(pypotential_a1[...,n_t//2], origin='lower', extent=sim.x_extent)
# plt.show()

# plt.imshow(pypotential_a2[...,n_t//2], origin='lower', extent=sim.x_extent)
# plt.show()

# plt.imshow(pypotential_a0[...,n_t//2], origin='lower', extent=sim.x_extent)
# plt.show()

###
### Run solver
###

blocks = 8
total_timesteps = n_t

t1 = time()
for iter in range(blocks):

    print("=== Block %i, current time: %i ===" %(iter, total_time//blocks * iter))

    sim = solver.kgsim(L,m,N2)

    t_span = (iter*total_time/blocks,(iter+1)*total_time/blocks)

    n_timesteps = total_timesteps//blocks
    n_t_span = (iter*n_timesteps, (iter+1)*n_timesteps)

    t0 = time()
    sim.solve_t(pypotential,pypotential_a1,pypotential_a2, pypotential_a0, phi_bar,t_span,n_t_span,n_timesteps,kgform='phibar')
    # sim.solve(pypotential,pypotential_a1,pypotential_a2, pypotential_a0, phi_bar,t_span,n_timesteps,kgform='phibar')
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

gif_id = "_time_pot%.2f_mom%.2f_"%(pot0,np.sqrt((px0)**2+(py0)**2))
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