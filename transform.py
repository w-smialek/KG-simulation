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
