import numpy as np
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
import pylab as plt
from scipy import integrate

m = 1
hbar = 1
c = 1

L = 20*hbar/(m*c)   # length of 1-sphere in x and y
N2 = 30            # max positive/negative mode in px and py
Ntot = 2*N2+1

def colorize(z):
    n,m = z.shape
    c = np.zeros((n,m,3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0 - 1.0/(1.0+abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a,b in zip(A,B)]
    return c

def ene(px,py):
    return np.sqrt(m**2 + px**2 + py**2)

def t_space_rot(nx,ny):
    px = hbar*2*np.pi/L*nx
    py = hbar*2*np.pi/L*ny
    U = 1/(2*np.sqrt(m*ene(px,py)))*np.array([[m+ene(px,py),m-ene(px,py)],[m-ene(px,py),m+ene(px,py)]])
    return U

def fv_to_field(phi_bar):
    phi = np.zeros(phi_bar.shape).astype(complex)
    psi = np.zeros(phi_bar.shape).astype(complex)
    for nx in range(-N2,N2+1):
        for ny in range(-N2,N2+1):
            phi[nx+N2,ny+N2,:] = t_space_rot(nx,ny)@phi_bar[nx+N2,ny+N2,:]
    for l in range(2):
        phi_shifted = np.roll(phi[:,:,l],(N2+1,N2+1),(0,1))
        psi[:,:,l] = Ntot**2*np.fft.ifft2(phi_shifted)
        psi[:,:,l] = np.fft.fftshift(psi[:,:,l])
    return 1/np.sqrt(2)*(psi[:,:,0]+psi[:,:,1])

p_extent = 2*np.pi*hbar/L*N2

space_nx, space_ny, space_l = np.meshgrid(range(-N2,N2+1),range(-N2,N2+1),range(2))
space_px, space_py, space_l = np.meshgrid(np.linspace(-p_extent,p_extent,Ntot),np.linspace(-p_extent,p_extent,Ntot),range(2))


phi_bar = np.zeros((Ntot,Ntot,2)).astype(complex)

nx0 = 10
ny0 = 20

phi_bar[N2+ny0,N2+nx0,0] = 1
phi_bar[N2+ny0,N2+nx0,1] = 0

b = 0.02 + 0j
nx0 = 3
ny0 = 5
kx = 5
ky = 5
# phi_bar = (space_l+1j)*np.exp(-b*((space_nx-nx0)**2 + (space_ny-ny0)**2) + 1j*10*b*(space_nx*kx+space_ny*ky))
# phi_in = np.zeros((Ntot,Ntot,2))
# phi_in = np.fromfunction(lambda x, y, l: l*np.exp(-b*((x-x0)**2 + (y-y0)**2) + 1j*b*(x*kx+y*ky)), (nx,ny,2))


phi = fv_to_field(phi_bar)

plt.imshow(colorize(phi_bar[:,:,0]), interpolation='none',extent=(-p_extent,p_extent,-p_extent,p_extent),origin='lower')
plt.show()
plt.imshow(colorize(phi_bar[:,:,1]), interpolation='none',extent=(-p_extent,p_extent,-p_extent,p_extent),origin='lower')
plt.show()
plt.imshow(colorize(phi[:,:]), interpolation='none',extent=(-L/2,L/2,-L/2,L/2),origin='lower')
plt.show()

plt.imshow(phi.real,interpolation='none',extent=(-L/2,L/2,-L/2,L/2),origin='lower')
plt.show()

def rhs(t,y):
    phi_arr = np.reshape(y,(Ntot,Ntot,2))
    retval = np.zeros((Ntot,Ntot,2)).astype(complex)
    for nx in range(-N2,N2+1):
        for ny in range(-N2,N2+1):
            px = hbar*2*np.pi/L*nx
            py = hbar*2*np.pi/L*ny
            hamil_val = ene(px,py)*np.array([[1,0],[0,-1]])*(-1j)
            retval[nx,ny,:] = hamil_val@phi_arr[nx,ny,:]
    return np.reshape(retval,(Ntot*Ntot*2,))

solution = integrate.solve_ivp(rhs,(0,1),np.reshape(phi_bar,(Ntot*Ntot*2,)),method='BDF')

filenames = []
filenames2 = []

for t in range(0,100):
    try:
        imarr = np.reshape(solution.y[...,t],(Ntot,Ntot,2))
        plt.imshow(colorize(imarr[:,:,1]), interpolation='none',extent=(0,Ntot,0,Ntot))
        plt.savefig('./ims/fig%i.png'%t,dpi=200)
        filenames.append('./ims/fig%i.png'%t)

        imarr_field = fv_to_field(imarr)
        plt.imshow(abs(imarr_field), interpolation='none',extent=(0,Ntot,0,Ntot))
        plt.savefig('./ims/f2ig%i.png'%t,dpi=200)
        filenames2.append('./ims/f2ig%i.png'%t)

    except:
        break


import imageio
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('anim.gif', images)
images = []
for filename in filenames2:
    images.append(imageio.imread(filename))
imageio.mimsave('anim2.gif', images)