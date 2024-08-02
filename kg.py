import numpy as np
import matplotlib.pyplot as plt
from fastccolor.colorize import colorize
from array import array
from cy_solver import solver
from ftinterp import ftinterp
# from cy_solver import phi_to_psi_interp_cy
from PIL import Image, ImageOps
from time import time
from scipy import interpolate
from barcy import barcy

N2 = 150            # max positive/negative mode in px and py
Ntot = 2*N2         # Total number of modes in any dimension

###
### Relations between program variables and physical variables
###
#
m = 0.2             # mass in multiples of m_e
L = 200              # torus half-diameter in x and y in multiples of hbar/(m_e * c)
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

# def phi_bar_to_phi(phi_bar):
#     phi = np.zeros(phi_bar.shape).astype(complex)
#     for iy in range(-N2,N2):
#         for jx in range(-N2,N2):
#             row = N2 + iy 
#             col = N2 + jx
#             phi[:,row,col] = t_space_rot(jx,iy)@phi_bar[:,row,col]
#     return phi

def phi_bar_to_phi(phi_bar):

    mplus = 1/(2*np.sqrt(m*ene(space_px[0,...],space_py[0,...])))*(m+ene(space_px[0,...],space_py[0,...]))
    mminus = 1/(2*np.sqrt(m*ene(space_px[0,...],space_py[0,...])))*(m-ene(space_px[0,...],space_py[0,...]))
    
    u_mats = np.zeros((2,2,Ntot,Ntot))
    u_mats[0,0,...] = mplus
    u_mats[0,1,...] = mminus
    u_mats[1,0,...] = mminus
    u_mats[1,1,...] = mplus

    phi = np.einsum('ijnm,jnm->inm',u_mats,phi_bar)

    return phi

def phi_bar_to_phi_cy(phi_bar):
    phi_bar = flatten_for_cy(phi_bar)

    phi_bar = barcy.barcy(phi_bar, L, m, N2)

    phi = cy_to_numpy(phi_bar)
    return phi

def phi_bar_to_phi_interp(phi_bar, factor):
    phi = np.zeros(phi_bar.shape).astype(complex)
    for iy in range(-N2*factor,N2*factor):
        for jx in range(-N2*factor,N2*factor):
            row = N2*factor + iy 
            col = N2*factor + jx
            phi[:,row,col] = t_space_rot(jx/factor,iy/factor)@phi_bar[:,row,col]
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

def phi_to_psi_interp(phi, spacef_jx, spacef_iy, factor):
    psi = np.zeros(phi.shape).astype(complex)

    for d in range(2):
        psi[d,...] = (Ntot*factor)**2*np.fft.ifft2(phi[d,...])*np.exp(1j*np.pi*(spacef_jx+spacef_iy))
        psi[d,...] = np.roll(psi[d,...],(N2*factor,N2*factor),(0,1))

    varphi, idtvarphi = 1/np.sqrt(2)*(psi[0,:,:]+psi[1,:,:]), 1/np.sqrt(2)*(psi[0,:,:]-psi[1,:,:])
    return varphi, idtvarphi

def phi_to_psi_interp2(phi, px_interp, py_interp, factor):
    psi = np.zeros(phi.shape).astype(complex)
    for iy in range(-N2*factor,N2*factor):      # ROWS ARE Y,       FROM ROW     0 -> iy = -N2 -> Y = -L
        for jx in range(-N2*factor,N2*factor):  # COLLUNMS ARE X,   FROM COLLUMN 0 -> jx = -N2 -> X = -L
            for d in range(2):                  # DEPTH IS l,       d = 0 -> l = 1;  d = 1 -> q = -1
                row = N2*factor + iy 
                col = N2*factor + jx
                xx = x(jx/factor)
                yy = x(iy/factor)

                ft_ar = phi[d,...] * np.exp(1j*(px_interp*xx + py_interp*yy))
                psi_xy = np.sum(ft_ar)
                psi[d,row,col] = psi_xy
        print(iy)
    varphi, idtvarphi = 1/np.sqrt(2)*(psi[0,:,:]+psi[1,:,:]), 1/np.sqrt(2)*(psi[0,:,:]-psi[1,:,:])
    return varphi, idtvarphi

# def phi_to_psi_interp2(phi, px_interp, py_interp, factor):

#     psi = np.zeros(phi.shape).astype(complex)
#     px2 = np.zeros(phi.shape).astype(complex)
#     py2 = np.zeros(phi.shape).astype(complex)

#     px2[0,...] = px_interp
#     px2[1,...] = px_interp
#     py2[0,...] = py_interp
#     py2[1,...] = py_interp

#     phi2 = flatten_for_cy_interp(phi, factor)
#     psi2 = flatten_for_cy_interp(psi, factor)
#     px2 = flatten_for_cy_interp(px2, factor)
#     py2 = flatten_for_cy_interp(py2, factor)

#     psiar = ftinterp(psi2, phi2, px2, py2, factor, N2, L)

#     psi = cy_to_numpy_interp(psiar, factor)

#     varphi, idtvarphi = 1/np.sqrt(2)*(psi[0,:,:]+psi[1,:,:]), 1/np.sqrt(2)*(psi[0,:,:]-psi[1,:,:])
#     return varphi, idtvarphi


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

    a_out_re = np.zeros((2,Ntot,Ntot)).astype(complex)
    a_out_im = np.zeros((2,Ntot,Ntot)).astype(complex)

    a_out_re0 = np.reshape(a[0::4],(Ntot,Ntot),order="C").astype(complex)
    a_out_im0 = np.reshape(a[1::4],(Ntot,Ntot),order="C").astype(complex)
    a_out_re1 = np.reshape(a[2::4],(Ntot,Ntot),order="C").astype(complex)
    a_out_im1 = np.reshape(a[3::4],(Ntot,Ntot),order="C").astype(complex)

    a_out_re[0,...] = a_out_re0
    a_out_re[1,...] = a_out_re1
    a_out_im[0,...] = a_out_im0
    a_out_im[1,...] = a_out_im1

    # for nx in range(Ntot):
    #     for ny in range(Ntot):
    #         for l in range(2):
    #             a_out_re[l,nx,ny] = a[nx*Ntot*2*2 + ny*2*2 + l*2 +0]
    #             a_out_im[l,nx,ny] = a[nx*Ntot*2*2 + ny*2*2 + l*2 +1]

    return a_out_re + 1j*a_out_im

def flatten_for_cy_interp(a,factor):
    '''Convert feshbach - villard representation 2d complex field into a 1D python array,
    with index = nx * (Ntot x 2 x 2) + ny * (2 x 2) + l * 2 + c '''

    a_re = a.real.astype(float)
    a_im = a.imag.astype(float)

    ntf = Ntot*factor

    a_out = np.zeros((2*2*ntf*ntf))

    for nx in range(ntf):
        for ny in range(ntf):
            for l in range(2):
                a_out[nx*ntf*2*2 + ny*2*2 + l*2 +0] = a_re[l,nx,ny]
                a_out[nx*ntf*2*2 + ny*2*2 + l*2 +1] = a_im[l,nx,ny]

    a_out = array('d',a_out)
    return a_out

def cy_to_numpy_interp(a,factor):
    '''inverse of flatten_for_cy'''

    ntf = Ntot*factor

    a_out_re = np.zeros((2,ntf,ntf)).astype(complex)
    a_out_im = np.zeros((2,ntf,ntf)).astype(complex)

    for nx in range(ntf):
        for ny in range(ntf):
            for l in range(2):
                a_out_re[l,nx,ny] = a[nx*ntf*2*2 + ny*2*2 + l*2 +0]
                a_out_im[l,nx,ny] = a[nx*ntf*2*2 + ny*2*2 + l*2 +1]

    return a_out_re + 1j*a_out_im

def complex_interp_phi(phi, py_interp, px_interp, factor):
    phi_interp = np.zeros((2,Ntot*factor,Ntot*factor)).astype(complex)
    intplt0re = interpolate.RegularGridInterpolator([p_linspace,p_linspace], phi[0,...].real)
    intplt0im = interpolate.RegularGridInterpolator([p_linspace,p_linspace], phi[0,...].imag)
    intplt1re = interpolate.RegularGridInterpolator([p_linspace,p_linspace], phi[1,...].real)
    intplt1im = interpolate.RegularGridInterpolator([p_linspace,p_linspace], phi[1,...].imag)

    phi_interp[0,...] = intplt0re((py_interp,px_interp)) + 1j*intplt0im((py_interp,px_interp))
    phi_interp[1,...] = intplt1re((py_interp,px_interp)) + 1j*intplt1im((py_interp,px_interp))
    return phi_interp

def complex_interp_varphi(varphi, y_interp, x_interp, factor): # ,idtvarphi):
    intplt0re = interpolate.RegularGridInterpolator([x_linspace,x_linspace], varphi.real)
    intplt0im = interpolate.RegularGridInterpolator([x_linspace,x_linspace], varphi.imag)
    # intplt1re = interpolate.RegularGridInterpolator([x_linspace,x_linspace], idtvarphi.real)
    # intplt1im = interpolate.RegularGridInterpolator([x_linspace,x_linspace], idtvarphi.imag)

    varphi_interp = intplt0re((y_interp,x_interp)) + 1j*intplt0im((y_interp,x_interp))
    # idtvarphi_interp = intplt1re((y_interp,x_interp)) + 1j*intplt1im((y_interp,x_interp))
    return varphi_interp#, idtvarphi_interp

def colorpy(arr):
    return

###
### Prepare initial conditions
###

print('Length: ',L)
print('p_extent: ',p_extent_hi)

x0 = L/5
y0 = -L/3
px0 = -p_extent_hi*(1/2)
py0 = p_extent_hi*(1/4)
a_gauss = 5
phi = (1+space_l)*np.exp(-1j*(x0*space_px + y0*space_py) - a_gauss*10/p_extent_hi*((space_px - px0)**2 + (space_py - py0)**2))
phi += (1-space_l)*np.exp(-1j*(x0*space_px + y0*space_py) - a_gauss*10/p_extent_hi*((space_px + px0)**2 + (space_py + py0)**2))

# phi = np.exp(-1j*(0*space_px + 0*space_py))


phi_bar = phi_to_phi_bar(phi)
phi_bar = phi

pb_array = flatten_for_cy(phi_bar)
coefs = array('d',[N2,L,m])

t_init = 0.
# t_end = 40.0  # around 15 seconds per 1.0 on N2 = 100
# n_timesteps = 100

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

factor = 2
stretch = 2
interpolate_phi_bar = False
fps = 50
cmap1 = plt.get_cmap('binary')
t0 = time()

p_linspace_interp = np.linspace(p_extent_lo,p_extent_hi,Ntot*factor)
x_linspace_interp = np.linspace(-L,L,Ntot*factor)
px_interp, py_interp = np.meshgrid(p_linspace_interp, p_linspace_interp, indexing='ij')
x_interp, y_interp = np.meshgrid(x_linspace_interp, x_linspace_interp, indexing='ij')
spacef_jx = np.zeros((factor*Ntot,factor*Ntot))
spacef_iy = np.zeros((factor*Ntot,factor*Ntot))
interp_range = range(-N2*factor,N2*factor)
for iy in interp_range:       # ROWS ARE Y,       FROM ROW     0 -> iy = -N2 -> Y = -L
    for jx in interp_range:   # COLLUNMS ARE X,   FROM COLLUMN 0 -> jx = -N2 -> X = -L
        row = N2 + iy 
        col = N2 + jx

        spacef_jx[row,col] = jx
        spacef_iy[row,col] = iy

imagesa = []
imagesb = []
imagesc = []
for i in range(n_timesteps):

    t00 = time()
    sol_phi_bar = cy_to_numpy(result.y[:,i])
    tee = time()
    print('transforms1 time: ',tee-t00)

    t00 = time()
    sol_phi = phi_bar_to_phi(sol_phi_bar)
    # sol_phi = phi_bar_to_phi_cy(sol_phi_bar)
    tee = time()
    print('transforms2 time: ',tee-t00)

    t00 = time()
    sol_varphi, sol_idtvarphi = phi_to_psi(sol_phi)
    tee = time()
    print('transforms3 time: ',tee-t00)

    t00 = time()
    if factor != 1:
        if interpolate_phi_bar:
            sol_phi_bar = complex_interp_phi(sol_phi_bar, py_interp, px_interp, factor)
        sol_varphi = complex_interp_varphi(sol_varphi, x_interp, y_interp, factor)
        # sol_phi_bar = complex_interp_phi(sol_phi_bar, py_interp, px_interp, factor)
        # sol_phi = phi_bar_to_phi_interp(sol_phi_bar, factor)#, spacef_jx, spacef_iy, factor
        # # sol_varphi = phi_to_psi_interp(sol_phi, spacef_jx, spacef_iy, factor)[0]
        # sol_varphi = phi_to_psi_interp2(sol_phi, px_interp, py_interp, factor)[0]
    tee = time()
    print('interpolations time: ',tee-t00)

    t00 = time()
    datac_phi_bar = colorize(sol_phi_bar[0,...], stretch)
    datac_varphi = colorize(sol_varphi, stretch)
    tee = time()
    print('coloring1 time: ',tee-t00)

    t00 = time()
    databs_varphi = abs(sol_varphi)
    databs_varphi = databs_varphi/(np.sum(databs_varphi))*np.power(Ntot*factor,2/1.25)
    databs_varphi = cmap1(databs_varphi)

    if stretch != 1:
        databs_varphi = np.repeat(np.repeat(databs_varphi,stretch, axis=0), stretch, axis=1)
    tee = time()
    print('coloring2 time: ',tee-t00)

    t00 = time()
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

    tee = time()
    print('saving images time: ',tee-t00)


    if (i%2==0):
        print(i)
    
gif_id = 5
imagesa[0].save("anim%ia.gif"%gif_id, save_all = True, append_images=imagesa[1:], duration = 1/fps*1000, loop=0)
imagesb[0].save("anim%ib.gif"%gif_id, save_all = True, append_images=imagesb[1:], duration = 1/fps*1000, loop=0)
imagesc[0].save("anim%ic.gif"%gif_id, save_all = True, append_images=imagesc[1:], duration = 1/fps*1000, loop=0)

te = time()
print("rendering images time: %f"%(te-t0))