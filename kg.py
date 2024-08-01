import numpy as np
import matplotlib.pyplot as plt
from fastccolor.colorize import colorize
from array import array
from cy_solver import solver
from PIL import Image, ImageOps
from time import time
from scipy import interpolate

N2 = 60            # max positive/negative mode in px and py
Ntot = 2*N2         # Total number of modes in any dimension

###
### Relations between program variables and physical variables
###
#
m = 1.0             # mass in multiples of m_e
L = 16              # torus half-diameter in x and y in multiples of hbar/(m_e * c)
#                   # (that means THE TOTAL LENGTH IS 2L)
#
# time_phys = time_var * hbar/(c^2 * m_e)
# p_phys    = p_var    * m_e * c
# ene_phys  = ene_var  * m_e * c^2
# x_phys    = x_var    * hbar/(m_e * c)
#
def p(n):
    return 2*np.pi/(2*L)*n
def x(n):
    return n/N2*L
#
### Explicit dimensions of the space
#
p_extent_lo = p(-N2)
p_extent_hi = p(N2-1)
#
p_extent = (p_extent_lo,p_extent_hi,p_extent_lo,p_extent_hi)
x_extent = (-L,L,-L,L)
#
p_linspace = np.linspace(p_extent_lo,p_extent_hi,Ntot)
x_linspace = np.linspace(-L,L,Ntot)
#
###
### Coordinates
###

space_jx = np.zeros((2,Ntot,Ntot))
space_iy = np.zeros((2,Ntot,Ntot))

space_px = np.zeros((2,Ntot,Ntot))
space_py = np.zeros((2,Ntot,Ntot))

space_x = np.zeros((2,Ntot,Ntot))
space_y = np.zeros((2,Ntot,Ntot))
space_l = np.zeros((2,Ntot,Ntot))

for iy in range(-N2,N2):      # ROWS ARE Y,       FROM ROW     0 -> iy = -N2 -> Y = -L
    for jx in range(-N2,N2):  # COLLUNMS ARE X,   FROM COLLUMN 0 -> jx = -N2 -> X = -L
        for d in range(2):    # DEPTH IS l,       d = 0 -> l = 1;  d = 1 -> q = -1
            row = N2 + iy 
            col = N2 + jx
            l = 1 - 2*d

            space_px[d,row,col] = p(jx)
            space_py[d,row,col] = p(iy)

            space_x[d,row,col] = x(jx)
            space_y[d,row,col] = x(iy)
            space_l[d,row,col] = l

            space_jx[d,row,col] = jx
            space_iy[d,row,col] = iy

###
### Transformations
###

def ene(px,py):
    return np.sqrt(m**2 + px**2 + py**2)

def t_space_rot(nx,ny):      # phi_p = U(p) @ phi_bar_p
    px = 2*np.pi/L*nx
    py = 2*np.pi/L*ny
    U = 1/(2*np.sqrt(m*ene(px,py)))*np.array([[m+ene(px,py),m-ene(px,py)],[m-ene(px,py),m+ene(px,py)]])
    return U

def t_space_rot_inv(nx,ny):  # phi_bar_p = U_inv(p) @ phi_p
    px = 2*np.pi/L*nx
    py = 2*np.pi/L*ny
    U = 1/(2*np.sqrt(m*ene(px,py)))*np.array([[m+ene(px,py),ene(px,py)-m],[ene(px,py)-m,m+ene(px,py)]])
    return U

def phi_bar_to_phi(phi_bar):
    phi = np.zeros(phi_bar.shape).astype(complex)
    for iy in range(-N2,N2):
        for jx in range(-N2,N2):
            row = N2 + iy 
            col = N2 + jx
            phi[:,row,col] = t_space_rot(jx,iy)@phi_bar[:,row,col]
    return phi

def phi_to_phi_bar(phi):
    phibar = np.zeros(phi.shape).astype(complex)
    for iy in range(-N2,N2):
        for jx in range(-N2,N2):
            row = N2 + iy 
            col = N2 + jx
            phibar[:,row,col] = t_space_rot_inv(jx,iy)@phi[:,row,col]
    return phibar

# def phi_to_psi(phi):
#     psi = np.zeros(phi.shape).astype(complex)
#     for iy in range(-N2,N2):      # ROWS ARE Y,       FROM ROW     0 -> iy = -N2 -> Y = -L
#         for jx in range(-N2,N2):  # COLLUNMS ARE X,   FROM COLLUMN 0 -> jx = -N2 -> X = -L
#             for d in range(2):    # DEPTH IS l,       d = 0 -> l = 1;  d = 1 -> q = -1
#                 row = N2 + iy 
#                 col = N2 + jx
#                 xx = x(jx)
#                 yy = x(iy)

#                 ft_ar = phi[d,...] * np.exp(1j*(space_px[d,...]*xx + space_py[d,...]*yy))
#                 psi_xy = np.sum(ft_ar)
#                 psi[d,row,col] = psi_xy
#     varphi, idtvarphi = 1/np.sqrt(2)*(psi[0,:,:]+psi[1,:,:]), 1/np.sqrt(2)*(psi[0,:,:]-psi[1,:,:])
#     return varphi, idtvarphi

def phi_to_psi(phi):
    psi = np.zeros(phi.shape).astype(complex)

    for d in range(2):
        psi[d,...] = Ntot**2*np.fft.ifft2(phi[d,...])*np.exp(1j*np.pi*(space_jx+space_iy)[0,...])
        psi[d,...] = np.roll(psi[d,...],(N2,N2),(0,1))

    varphi, idtvarphi = 1/np.sqrt(2)*(psi[0,:,:]+psi[1,:,:]), 1/np.sqrt(2)*(psi[0,:,:]-psi[1,:,:])
    return varphi, idtvarphi

def psi_to_phi(varphi, idtvarphi):
    phi = np.zeros((2,Ntot,Ntot)).astype(complex)
    psi = np.zeros((2,Ntot,Ntot)).astype(complex)
    psi[0,...] = 1/np.sqrt(2)*(varphi+idtvarphi)
    psi[1,...] = 1/np.sqrt(2)*(varphi-idtvarphi)

    for d in range(2):
        psi[d,...] = np.roll(psi[d,...],(-N2,-N2),(0,1))
        phi[d,...] = 1/Ntot**2*np.fft.fft2(psi[d,...])
        phi[d,...] = np.roll(phi[d,...],(N2,N2),(0,1))

    return phi

# for i in range(-N2,N2):
#     for j in range(-N2,N2):
#         plane = np.zeros(space_l.shape)
#         plane[0,N2+i,N2+j] = 1
        
#         plane = np.exp(-1j*(x(j)*space_px + x(i)*space_py))*(1+space_l)
#         plane[0,N2+i,N2+j] = 1

#         var, idt = phi_to_psi(plane)
#         plane_rev = psi_to_phi(var, idt)

#         if np.any(np.abs(plane-plane_rev) > 1.0e-10):
#             print("error!", i, j)
#             print(np.max(np.abs(plane-plane_rev)))
#             plt.imshow(colorize(plane[0,...]), origin='lower', extent=x_extent)
#             plt.show()
#             plt.imshow(colorize(plane_rev[0,...]), origin='lower', extent=x_extent)
#             plt.show()
#             plt.imshow(np.sum(np.abs(plane - plane_rev),0), origin='lower', extent=x_extent)
#             plt.show()
#         else:
#             print("ok", i, j)

def flatten_for_cy(a):
    '''Convert feshbach - villard representation 2d complex field into a 1D python array,
    with index = nx * (Ntot x 2 x 2) + ny * (2 x 2) + l * 2 + c '''

    a_re = a.real.astype(float)
    a_im = a.imag.astype(float)

    a_out = np.zeros((2*2*Ntot*Ntot))

    for nx in range(Ntot):
        for ny in range(Ntot):
            for l in range(2):
                a_out[nx*Ntot*2*2 + ny*2*2 + l*2 +0] = a_re[l,nx,ny]
                a_out[nx*Ntot*2*2 + ny*2*2 + l*2 +1] = a_im[l,nx,ny]

    # a_out = a.reshape((-1,),order='F')
    # a_out = a_out.view(float).reshape((-1,),order='F')
    a_out = array('d',a_out)
    return a_out

def cy_to_numpy(a):
    '''inverse of flatten_for_cy'''
    # a_out_re = np.reshape(a[0::2],(2,Ntot,Ntot),order="F").astype(complex)
    # a_out_im = np.reshape(a[1::2],(2,Ntot,Ntot),order="F").astype(complex)

    a_out_re = np.zeros((2,Ntot,Ntot)).astype(complex)
    a_out_im = np.zeros((2,Ntot,Ntot)).astype(complex)

    for nx in range(Ntot):
        for ny in range(Ntot):
            for l in range(2):
                a_out_re[l,nx,ny] = a[nx*Ntot*2*2 + ny*2*2 + l*2 +0]
                a_out_im[l,nx,ny] = a[nx*Ntot*2*2 + ny*2*2 + l*2 +1]

    return a_out_re + 1j*a_out_im

def complex_interp_phi(phi, py_interp, px_interp):
    phi_interp = np.zeros(phi.shape)
    intplt0re = interpolate.RegularGridInterpolator([p_linspace,p_linspace], phi[0,...].real)
    intplt0im = interpolate.RegularGridInterpolator([p_linspace,p_linspace], phi[0,...].imag)
    intplt1re = interpolate.RegularGridInterpolator([p_linspace,p_linspace], phi[1,...].real)
    intplt1im = interpolate.RegularGridInterpolator([p_linspace,p_linspace], phi[1,...].imag)

    phi_interp[0,...] = intplt0re((py_interp,px_interp)) + 1j*intplt0im((py_interp,px_interp))
    phi_interp[1,...] = intplt1re((py_interp,px_interp)) + 1j*intplt1im((py_interp,px_interp))

    return phi_interp

###
### Prepare initial conditions
###

x0 = 1.4
y0 = -2.5
phi = np.exp(-1j*(x0*space_px + y0*space_py))

phi_bar = phi_to_phi_bar(phi)

pb_array = flatten_for_cy(phi_bar)
coefs = array('d',[N2,L,m])

t_init = 0.
t_end = 2.0  # around 15 seconds per 1.0 on N2 = 100
n_timesteps = 10

t_span = (t_init, t_end)
timesteps = array('d',np.linspace(t_init, t_end, n_timesteps))

###
### Run solver
###

t0 = time()
result = solver(t_span, pb_array, coefs, timesteps)
te = time()
print("Mycyrk time: %f"%(te-t0))

print(result.message)
print("Size of solution: ", result.size)


###
### Render pictures
###

factor = 1
stretch = 2
fps = 50
cmap1 = plt.get_cmap('binary')
t0 = time()

p_linspace_interp = np.linspace(p_extent_lo,p_extent_hi,Ntot*factor)
py_interp, px_interp = np.meshgrid(p_linspace_interp, p_linspace_interp, indexing='ij')

imagesa = []
imagesb = []
imagesc = []
for i in range(n_timesteps):

    sol_phi_bar = cy_to_numpy(result.y[:,i])
    if factor != 1:
        sol_phi_bar = complex_interp_phi(sol_phi_bar, py_interp, px_interp)
    
    sol_varphi = phi_to_psi(phi_bar_to_phi(sol_phi_bar))[0]

    datac_phi_bar = colorize(sol_phi_bar[0,...], stretch)
    datac_varphi = colorize(sol_varphi, stretch)

    databs_varphi = abs(sol_varphi)
    databs_varphi = databs_varphi/np.max(databs_varphi)
    databs_varphi = cmap1(databs_varphi)
    if stretch != 1:
        databs_varphi = np.repeat(np.repeat(databs_varphi,stretch, axis=0), stretch, axis=1)

    imga = Image.fromarray((datac_phi_bar[:, :, :3] * 255).astype(np.uint8))
    imga = ImageOps.flip(imga)
    imga.save('./ims/afig%i.png'%i)
    imagesa.append(imga)

    imgb = Image.fromarray((datac_varphi[:, :, :3] * 255).astype(np.uint8))
    imgb = ImageOps.flip(imgb)
    imgb.save('./ims/bfig%i.png'%i)
    imagesb.append(imgb)

    imgc = Image.fromarray((databs_varphi[:, :, :3] * 255).astype(np.uint8))
    imgc = ImageOps.flip(imgc)
    imgc.save('./ims/cfig%i.png'%i)
    imagesc.append(imgc)

    if (i%20==0):
        print(i)
    
imagesa[0].save("anima.gif", save_all = True, append_images=imagesa[1:], duration = 1/fps*1000, loop=0)
imagesb[0].save("animb.gif", save_all = True, append_images=imagesb[1:], duration = 1/fps*1000, loop=0)
imagesc[0].save("animc.gif", save_all = True, append_images=imagesc[1:], duration = 1/fps*1000, loop=0)

# images = []
# for i in range(n_timesteps):
#     datac = colorize(phi_to_psi(phi_bar_to_phi(cy_to_numpy(result.y[:,i])))[0], stretch)
#     img = Image.fromarray((datac[:, :, :3] * 255).astype(np.uint8))
#     img = ImageOps.flip(img)
#     img.save('./ims/afig%i.png'%i)
#     images.append(img)
#     if (i%20==0):
#         print(i)
# images[0].save("anim2.gif", save_all = True, append_images=images[1:], duration = 1/fps*1000, loop=0)


# images = []
# for i in range(n_timesteps):
#     datac = abs(phi_to_psi(phi_bar_to_phi(cy_to_numpy(result.y[:,i])))[0])
#     datac = datac/np.max(datac)
#     datac = cmap1(datac)
#     datac = np.repeat(np.repeat(datac,stretch, axis=0), stretch, axis=1)

#     img = Image.fromarray((datac[:, :, :3] * 255).astype(np.uint8))
#     img = ImageOps.flip(img)
#     img.save('./ims/bfig%i.png'%i)
#     images.append(img)
#     if (i%20==0):
#         print(i)
# images[0].save("anim3.gif", save_all = True, append_images=images[1:], duration = 1/fps*1000, loop=0)

te = time()
print("rendering images time: %f"%(te-t0))

###
### Interpolation
###

data = cy_to_numpy(result.y[:,5])
plt.imshow(colorize(data[0,...]))
plt.show()

p_linspace = np.linspace(p_extent_lo,p_extent_hi,Ntot)

intplt0re = interpolate.RegularGridInterpolator([p_linspace,p_linspace], data[0,...].real)
intplt0im = interpolate.RegularGridInterpolator([p_linspace,p_linspace], data[0,...].imag)
intplt1re = interpolate.RegularGridInterpolator([p_linspace,p_linspace], data[1,...].real)
intplt1im = interpolate.RegularGridInterpolator([p_linspace,p_linspace], data[1,...].imag)

factor = 4
p_linspace_interp = np.linspace(p_extent_lo,p_extent_hi,Ntot*factor)
py_interp, px_interp = np.meshgrid(p_linspace_interp, p_linspace_interp, indexing='ij')

datan = intplt0re([py_interp,px_interp]) + 1j*intplt0im([py_interp,px_interp])

print(datan.shape)

plt.imshow(colorize(datan))
plt.show()
