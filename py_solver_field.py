import numpy as np
from scipy import interpolate
from fastccolor.colorize import colorize
from array import array
import matplotlib.pyplot as plt
from cy_solver_field import solver
from PIL import Image, ImageOps, ImageDraw
import cv2

class kgsim:

    def __init__(self, l, m, n2):
        self.l = l
        self.m = m
        self.n2 = n2
        if n2 > 150:
            raise NotImplementedError("maximum lattice size is currently 300x300")
        self.ntot = 2*n2
        self.p = lambda n: 2*np.pi/(2*l)*n
        self.x = lambda n: l*n/n2
        self.ene = lambda px, py: np.sqrt(m**2 + px**2 + py**2)

        self.p_extent_lo = self.p(-n2)
        self.p_extent_hi = self.p(n2-1)
        self.x_extent_lo = self.x(-n2)
        self.x_extent_hi = self.x(n2-1)

        self.p_extent = (self.p_extent_lo,self.p_extent_hi,self.p_extent_lo,self.p_extent_hi)
        self.x_extent = (self.x_extent_lo,self.x_extent_hi,self.x_extent_lo,self.x_extent_hi)

        self.p_linspace = np.linspace(self.p_extent_lo,self.p_extent_hi,self.ntot)
        self.x_linspace = np.linspace(self.x_extent_lo,self.x_extent_hi,self.ntot)

        self.space_l, self.space_iy, self.space_jx = np.meshgrid([1,-1], range(-n2,n2),range(-n2,n2), indexing='ij')
        _, self.space_py, self.space_px = np.meshgrid([1,-1], self.p_linspace, self.p_linspace, indexing='ij')
        _, self.space_y, self.space_x = np.meshgrid([1,-1], self.x_linspace, self.x_linspace, indexing='ij')

        self.loaded = []

    def t_space_rot(self,nx,ny):      # phi_p = U(p) @ phi_bar_p
        px = 2*np.pi/self.l*nx
        py = 2*np.pi/self.l*ny
        U = 1/(2*np.sqrt(self.m*self.ene(px,py)))*np.array([[self.m+self.ene(px,py),self.m-self.ene(px,py)],[self.m-self.ene(px,py),self.m+self.ene(px,py)]])
        return U

    def t_space_rot_inv(self,nx,ny):  # phi_bar_p = U_inv(p) @ phi_p
        px = 2*np.pi/self.l*nx
        py = 2*np.pi/self.l*ny
        U = 1/(2*np.sqrt(self.m*self.ene(px,py)))*np.array([[self.m+self.ene(px,py),self.ene(px,py)-self.m],[self.ene(px,py)-self.m,self.m+self.ene(px,py)]])
        return U
    
    def repr_transform(self,in_field,forms):
        in_form,_,out_form = forms.partition('->')
        match in_form:
            case 'phibar':
                in_form = 0
            case 'phi':
                in_form = 1
            case 'psi':
                in_form = 2
            case 'varphi':
                in_form = 3
            case _: 
                raise NotImplementedError("wrong field form")
            
        match out_form:
            case 'phibar':
                in_form = 0
            case 'phi':
                in_form = 1
            case 'psi':
                in_form = 2
            case 'varphi':
                in_form = 3
            case _: 
                raise NotImplementedError("wrong field form")
            
        


    def phi_bar_to_phi(self,phi_bar):

        mplus = 1/(2*np.sqrt(self.m*self.ene(self.space_px[0,...],self.space_py[0,...])))*(self.m+self.ene(self.space_px[0,...],self.space_py[0,...]))
        mminus = 1/(2*np.sqrt(self.m*self.ene(self.space_px[0,...],self.space_py[0,...])))*(self.m-self.ene(self.space_px[0,...],self.space_py[0,...]))
        
        u_mats = np.zeros((2,2,self.ntot,self.ntot))
        u_mats[0,0,...] = mplus
        u_mats[0,1,...] = mminus
        u_mats[1,0,...] = mminus
        u_mats[1,1,...] = mplus

        phi = np.einsum('ijnm,jnm->inm',u_mats,phi_bar)

        return phi

    def phi_to_phi_bar(self,phi):
        phibar = np.zeros(phi.shape).astype(complex)
        for iy in range(-self.n2,self.n2):
            for jx in range(-self.n2,self.n2):
                row = self.n2 + iy 
                col = self.n2 + jx
                phibar[:,row,col] = self.t_space_rot_inv(jx,iy)@phi[:,row,col]
        return phibar

    def phi_to_psi(self,phi):
        psi = np.zeros(phi.shape).astype(complex)

        for d in range(2):
            psi[d,...] = self.ntot**2*np.fft.ifft2(phi[d,...])*np.exp(1j*np.pi*(self.space_jx+self.space_iy)[0,...])
            psi[d,...] = np.roll(psi[d,...],(self.n2,self.n2),(0,1))

        varphi, idtvarphi = 1/np.sqrt(2)*(psi[0,:,:]+psi[1,:,:]), 1/np.sqrt(2)*(psi[0,:,:]-psi[1,:,:])
        return varphi, idtvarphi

    def psi_to_rho(self,varphi, idtvarphi):

        n,m = varphi.shape

        psi = np.zeros((2,n,m)).astype(complex)
        psi[0,...] = 1/np.sqrt(2)*(varphi+idtvarphi)
        psi[1,...] = 1/np.sqrt(2)*(varphi-idtvarphi)

        rho = np.abs(psi[0,...])**2 - np.abs(psi[1,...])**2
        
        return rho

    def psi_to_phi(self,psi):
        phi = np.zeros((2,self.ntot,self.ntot)).astype(complex)
        # psi = np.zeros((2,self.ntot,self.ntot)).astype(complex)
        # psi[0,...] = 1/np.sqrt(2)*(varphi+idtvarphi)
        # psi[1,...] = 1/np.sqrt(2)*(varphi-idtvarphi)

        for d in range(2):
            psi[d,...] = np.roll(psi[d,...],(-self.n2,-self.n2),(0,1))
            phi[d,...] = 1/self.ntot**2*np.fft.fft2(psi[d,...])
            phi[d,...] = np.roll(phi[d,...],(self.n2,self.n2),(0,1))

        return phi

    def flatten_for_cy(self,a):
        '''Convert feshbach - villard representation 2d complex field into a 1D python array,
        with index = nx * (Ntot x 2 x 2) + ny * (2 x 2) + l * 2 + c '''

        a_re = a.real.astype(float)
        a_im = a.imag.astype(float)

        a_out = np.zeros((2*2*self.ntot*self.ntot))

        for nx in range(self.ntot):
            for ny in range(self.ntot):
                for l in range(2):
                    a_out[nx*self.ntot*2*2 + ny*2*2 + l*2 +0] = a_re[l,nx,ny]
                    a_out[nx*self.ntot*2*2 + ny*2*2 + l*2 +1] = a_im[l,nx,ny]

        a_out = array('d',a_out)
        return a_out

    def flatten_for_cycol(self,a, n, m):
        '''Convert feshbach - villard representation 2d complex field into a 1D python array,
        with index = nx * (Ntot x 2 x 2) + ny * (2 x 2) + l * 2 + c '''

        a_out = a.reshape((-1,),order='C')
        a_out = a_out.view(float).reshape((-1,),order='C')
        a_out = array('d',a_out)
        return a_out

    def cycol_rgb_to_np(self,arr, n, m):
        c_out = np.zeros((n,m,3))

        c_out[...,0] = np.reshape(arr[0::3],(n,m),order="C")
        c_out[...,1] = np.reshape(arr[1::3],(n,m),order="C")
        c_out[...,2] = np.reshape(arr[2::3],(n,m),order="C")

        return c_out

    def cy_to_numpy(self,a):
        '''inverse of flatten_for_cy'''

        a_out_re = np.zeros((2,self.ntot,self.ntot)).astype(complex)
        a_out_im = np.zeros((2,self.ntot,self.ntot)).astype(complex)

        a_out_re0 = np.reshape(a[0::4],(self.ntot,self.ntot),order="C").astype(complex)
        a_out_im0 = np.reshape(a[1::4],(self.ntot,self.ntot),order="C").astype(complex)
        a_out_re1 = np.reshape(a[2::4],(self.ntot,self.ntot),order="C").astype(complex)
        a_out_im1 = np.reshape(a[3::4],(self.ntot,self.ntot),order="C").astype(complex)

        a_out_re[0,...] = a_out_re0
        a_out_re[1,...] = a_out_re1
        a_out_im[0,...] = a_out_im0
        a_out_im[1,...] = a_out_im1

        return a_out_re + 1j*a_out_im

    def complex_interp_phi(self,phi, py_interp, px_interp, factor):
        phi_interp = np.zeros((2,self.ntot*factor,self.ntot*factor)).astype(complex)
        intplt0re = interpolate.RegularGridInterpolator([self.p_linspace,self.p_linspace], phi[0,...].real)
        intplt0im = interpolate.RegularGridInterpolator([self.p_linspace,self.p_linspace], phi[0,...].imag)
        intplt1re = interpolate.RegularGridInterpolator([self.p_linspace,self.p_linspace], phi[1,...].real)
        intplt1im = interpolate.RegularGridInterpolator([self.p_linspace,self.p_linspace], phi[1,...].imag)

        phi_interp[0,...] = intplt0re((py_interp,px_interp)) + 1j*intplt0im((py_interp,px_interp))
        phi_interp[1,...] = intplt1re((py_interp,px_interp)) + 1j*intplt1im((py_interp,px_interp))
        return phi_interp

    def complex_interp_varphi(self,varphi, y_interp, x_interp, factor): # ,idtvarphi):
        intplt0re = interpolate.RegularGridInterpolator([self.x_linspace,self.x_linspace], varphi.real)
        intplt0im = interpolate.RegularGridInterpolator([self.x_linspace,self.x_linspace], varphi.imag)
        # intplt1re = interpolate.RegularGridInterpolator([x_linspace,x_linspace], idtvarphi.real)
        # intplt1im = interpolate.RegularGridInterpolator([x_linspace,x_linspace], idtvarphi.imag)

        varphi_interp = intplt0re((y_interp,x_interp)) + 1j*intplt0im((y_interp,x_interp))
        return varphi_interp#, idtvarphi_interp

    def colorpy(self,z):
        n,m = z.shape

        zar = self.flatten_for_cycol(z,n,m)

        car = array('d',np.zeros((n*m*3,)))

        car = colorize(zar, car, n, m)

        c_out = self.cycol_rgb_to_np(car,n,m)

        return c_out
    
    def psi_to_phi_pot(self,psi):
        phi = np.zeros((self.ntot,self.ntot)).astype(complex)
        psi = np.roll(psi,(-self.n2,-self.n2),(0,1))
        phi = 1/self.ntot**2*np.fft.fft2(psi)
        phi = np.roll(phi,(self.n2,self.n2),(0,1))
        return phi
    
    def flatten_for_cy_pot(self,c_field):
        potential_lst = 2*self.ntot*self.ntot*[0.]
        for i1 in range(self.ntot):
            for i2 in range(self.ntot):
                potential_lst[i1*self.ntot*2 + i2*2] = c_field[i1,i2].real
                potential_lst[i1*self.ntot*2 + i2*2 + 1] = c_field[i1,i2].imag

        potential_arr = array('d', potential_lst)
        return potential_arr
    
    def solve(self, potfield, kgfield, t_span, n_timesteps, kgform = 'phibar'):
        if kgform == 'phi':
            kgfield = self.phi_to_phi_bar(kgfield)
        elif kgform == 'psi':
            kgfield = self.psi_to_phi(kgfield)
            kgfield = self.phi_to_phi_bar(kgfield)
        elif kgform == 'phibar':
            pass
        else:
            print("Incorrect name of the kg field form")
    
        kgarr = self.flatten_for_cy(kgfield)

        # Default potential form is position space, 
        # I guess no need to consider other input form
        potfield = self.psi_to_phi_pot(potfield)
        potarr = self.flatten_for_cy_pot(potfield)

        t_init, t_end = t_span
        self.time_tot = t_end - t_init

        timesteps = array('d',np.linspace(t_init, t_end, n_timesteps))
        self.timesteps = np.linspace(t_init, t_end, n_timesteps)
        coefs = array('d',[self.n2,self.m,self.l])
        self.result = solver(t_span, kgarr, coefs, timesteps, potarr)      # around 15 seconds per 1.0 on N2 = 100
        self.n_timesteps = n_timesteps
        return
    
    def save(self,path,destroy_cyrk=False):
        savedarr = np.zeros((2,self.ntot,self.ntot,self.n_timesteps)).astype(complex)
        for i in range(self.n_timesteps):
            savedarr[:,:,:,i] = self.cy_to_numpy(self.result.y[:,i])
        np.save(path,savedarr)
        if destroy_cyrk:
            self.result = 0
        return savedarr[...,-1]

    def load(self,path):
        loadedarr = np.load(path)
        self.loaded.append(loadedarr)
    
    # def render(self,
    #            factor=1,
    #            fps=26,
    #            gif_id=1,
    #            pb_complex_plot=False,
    #            vp_complex_plot=False,
    #            vp_abs_plot=False,
    #            charge_plot=True,
    #            cmap_str='seismic',
    #            charge_satur_val=0,
    #            fromloaded=False):
        
    #     p_linspace_interp = np.linspace(self.p_extent_lo,self.p_extent_hi,self.ntot*factor)
    #     x_linspace_interp = np.linspace(self.x_extent_lo,self.x_extent_hi,self.ntot*factor)
    #     px_interp, py_interp = np.meshgrid(p_linspace_interp, p_linspace_interp, indexing='ij')
    #     x_interp, y_interp = np.meshgrid(x_linspace_interp, x_linspace_interp, indexing='ij')
    #     spacef_jx = np.zeros((factor*self.ntot,factor*self.ntot))
    #     spacef_iy = np.zeros((factor*self.ntot,factor*self.ntot))
    #     interp_range = range(-self.n2*factor,self.n2*factor)
    #     for iy in interp_range:       # ROWS ARE Y,       FROM ROW     0 -> iy = -N2 -> Y = -L
    #         for jx in interp_range:   # COLLUNMS ARE X,   FROM COLLUMN 0 -> jx = -N2 -> X = -L
    #             row = self.n2 + iy 
    #             col = self.n2 + jx

    #             spacef_jx[row,col] = jx
    #             spacef_iy[row,col] = iy

    #     cmap1 = plt.get_cmap(cmap_str)

    #     imagesa = []
    #     imagesb = []
    #     imagesc = []
    #     imagesd = []

    #     if fromloaded.any():
    #         self.loaded = np.concatenate(self.loaded,axis=3)
    #         load_tsteps = fromloaded
    #         self.n_timesteps = len(fromloaded)

    #     for i in range(self.n_timesteps):

    #         if fromloaded.any():
    #             sol_phi_bar = self.loaded[...,i]
    #         else:
    #             sol_phi_bar = self.cy_to_numpy(self.result.y[:,i])

    #         sol_phi = self.phi_bar_to_phi(sol_phi_bar)

    #         sol_varphi, sol_idtvarphi = self.phi_to_psi(sol_phi)

    #         if factor != 1:
    #             if pb_complex_plot:
    #                 sol_phi_bar_i = self.complex_interp_phi(sol_phi_bar, py_interp, px_interp, factor)
    #             if vp_complex_plot or vp_abs_plot or charge_plot:
    #                 sol_varphi = self.complex_interp_varphi(sol_varphi, x_interp, y_interp, factor)
    #             if charge_plot:
    #                 sol_idtvarphi = self.complex_interp_varphi(sol_idtvarphi, x_interp, y_interp, factor)

    #         if pb_complex_plot: ## THAT IS ONLY ONE CHARGE SIGN, ADD BOTH!
    #             datac_phi_bar = self.colorpy(sol_phi_bar_i[1,...])

    #             imga = Image.fromarray((datac_phi_bar[:, :, :3] * 255).astype(np.uint8))
    #             imga = ImageOps.flip(imga)
    #             imga.save('./ims/afig%i.png'%i)
    #             imagesa.append(imga)

    #         if vp_complex_plot:
    #             datac_varphi = self.colorpy(sol_varphi)

    #             imgb = Image.fromarray((datac_varphi[:, :, :3] * 255).astype(np.uint8))
    #             imgb = ImageOps.flip(imgb)
    #             imgb.save('./ims/bfig%i.png'%i)
    #             imagesb.append(imgb)


    #         if vp_abs_plot:
    #             databs_varphi = abs(sol_varphi)
    #             databs_varphi = databs_varphi/(np.sum(databs_varphi))*np.power(self.ntot*factor,2/1.25)
    #             databs_varphi = cmap1(databs_varphi)

    #             imgc = Image.fromarray((databs_varphi[:, :, :3] * 255).astype(np.uint8))
    #             imgc = ImageOps.flip(imgc)
    #             imgc.save('./ims/cfig%i.png'%i)
    #             imagesc.append(imgc)
    #             # print(np.sum(abs(sol_varphi)))

    #         if charge_plot:
    #             data_charge = self.psi_to_rho(sol_varphi, sol_idtvarphi)
    #             if charge_satur_val != 0:
    #                 data_charge_c = data_charge/charge_satur_val/2
    #                 data_charge_c += 1/2
    #             else:
    #                 data_charge_c = data_charge/(5*np.average(np.abs(data_charge)))/2
    #                 data_charge_c += 1/2
    #             data_charge_c = cmap1(data_charge_c)

    #             qdata = self.charges(sol_phi_bar)

    #             imgd = Image.fromarray((data_charge_c[:, :, :3] * 255).astype(np.uint8))
    #             imgd = ImageOps.flip(imgd)
    #             Idraw = ImageDraw.Draw(imgd)
    #             Idraw.text((20,20),
    #                         "Q+: %.4f\nQ-: %.4f\nQtotal: %.4f"%qdata,
    #                         fill=(0,0,0))
    #             if fromloaded.any():
    #                 Idraw.text((self.ntot*factor-75,20),
    #                             "time: %.2f"%load_tsteps[i],
    #                             fill=(0,0,0))
    #             else:
    #                 Idraw.text((self.ntot*factor-75,20),
    #                             "time: %.2f"%self.timesteps[i],
    #                             fill=(0,0,0))
    #             imgd.save('./ims/dfig%i.png'%i)
    #             imagesd.append(imgd)

    #         if (i%20==0):
    #             # print('average absolute charge: ',np.average(np.abs(data_charge)))
    #             print(i)

    #     if pb_complex_plot:
    #         imagesa[0].save("./gifs/anim%sa.gif"%gif_id, save_all = True, append_images=imagesa[1:], duration = 1/fps*1000, loop=0)
    #     if vp_complex_plot:
    #         imagesb[0].save("./gifs/anim%sb.gif"%gif_id, save_all = True, append_images=imagesb[1:], duration = 1/fps*1000, loop=0)
    #     if vp_abs_plot:
    #         imagesc[0].save("./gifs/anim%sc.gif"%gif_id, save_all = True, append_images=imagesc[1:], duration = 1/fps*1000, loop=0)
    #     if charge_plot:
    #         imagesd[0].save("./gifs/anim%sd.gif"%gif_id, save_all = True, append_images=imagesd[1:], duration = 1/fps*1000, loop=0)

    #     return

    def render(self,
               factor=1,
               fps=26,
               gif_id=1,
               pb_complex_plot=False,
               vp_complex_plot=False,
               vp_abs_plot=False,
               charge_plot=True,
               cmap_str='seismic',
               charge_satur_val=0,
               fromloaded=False):

        p_linspace_interp = np.linspace(self.p_extent_lo,self.p_extent_hi,self.ntot*factor)
        x_linspace_interp = np.linspace(self.x_extent_lo,self.x_extent_hi,self.ntot*factor)
        px_interp, py_interp = np.meshgrid(p_linspace_interp, p_linspace_interp, indexing='ij')
        x_interp, y_interp = np.meshgrid(x_linspace_interp, x_linspace_interp, indexing='ij')
        spacef_jx = np.zeros((factor*self.ntot,factor*self.ntot))
        spacef_iy = np.zeros((factor*self.ntot,factor*self.ntot))
        interp_range = range(-self.n2*factor,self.n2*factor)
        for iy in interp_range:       # ROWS ARE Y,       FROM ROW     0 -> iy = -N2 -> Y = -L
            for jx in interp_range:   # COLLUNMS ARE X,   FROM COLLUMN 0 -> jx = -N2 -> X = -L
                row = self.n2 + iy 
                col = self.n2 + jx

                spacef_jx[row,col] = jx
                spacef_iy[row,col] = iy

        cmap1 = plt.get_cmap(cmap_str)

        videodims = (self.ntot*factor,self.ntot*factor)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')    

        if pb_complex_plot:
            videoa = cv2.VideoWriter("./gifs/anim%sa.mp4"%gif_id,fourcc,fps,videodims)
        if vp_complex_plot:
            videob = cv2.VideoWriter("./gifs/anim%sb.mp4"%gif_id,fourcc,fps,videodims)
        if vp_abs_plot:
            videoc = cv2.VideoWriter("./gifs/anim%sc.mp4"%gif_id,fourcc,fps,videodims)
        if charge_plot:
            videod = cv2.VideoWriter("./gifs/anim%sd.mp4"%gif_id,fourcc,fps,videodims)

        if fromloaded.any():
            self.loaded = np.concatenate(self.loaded,axis=3)
            load_tsteps = fromloaded
            self.n_timesteps = len(fromloaded)

        for i in range(self.n_timesteps):

            if fromloaded.any():
                sol_phi_bar = self.loaded[...,i]
            else:
                sol_phi_bar = self.cy_to_numpy(self.result.y[:,i])

            sol_phi = self.phi_bar_to_phi(sol_phi_bar)

            sol_varphi, sol_idtvarphi = self.phi_to_psi(sol_phi)

            if factor != 1:
                if pb_complex_plot:
                    sol_phi_bar_i = self.complex_interp_phi(sol_phi_bar, py_interp, px_interp, factor)
                if vp_complex_plot or vp_abs_plot or charge_plot:
                    sol_varphi = self.complex_interp_varphi(sol_varphi, x_interp, y_interp, factor)
                if charge_plot:
                    sol_idtvarphi = self.complex_interp_varphi(sol_idtvarphi, x_interp, y_interp, factor)

            if pb_complex_plot: ## THAT IS ONLY ONE CHARGE SIGN, ADD BOTH!
                datac_phi_bar = self.colorpy(sol_phi_bar_i[1,...])

                imga = Image.fromarray((datac_phi_bar[:, :, :3] * 255).astype(np.uint8))
                imga = ImageOps.flip(imga)
                videoa.write(cv2.cvtColor(np.array(imga), cv2.COLOR_RGB2BGR))                

            if vp_complex_plot:
                datac_varphi = self.colorpy(sol_varphi)

                imgb = Image.fromarray((datac_varphi[:, :, :3] * 255).astype(np.uint8))
                imgb = ImageOps.flip(imgb)
                videob.write(cv2.cvtColor(np.array(imgb), cv2.COLOR_RGB2BGR))                

            if vp_abs_plot:
                databs_varphi = abs(sol_varphi)
                databs_varphi = databs_varphi/(np.sum(databs_varphi))*np.power(self.ntot*factor,2/1.25)
                databs_varphi = cmap1(databs_varphi)

                imgc = Image.fromarray((databs_varphi[:, :, :3] * 255).astype(np.uint8))
                imgc = ImageOps.flip(imgc)
                videoc.write(cv2.cvtColor(np.array(imgc), cv2.COLOR_RGB2BGR))                

            if charge_plot:
                data_charge = self.psi_to_rho(sol_varphi, sol_idtvarphi)
                if charge_satur_val != 0:
                    data_charge_c = data_charge/charge_satur_val/2
                    data_charge_c += 1/2
                else:
                    data_charge_c = data_charge/(5*np.average(np.abs(data_charge)))/2
                    data_charge_c += 1/2
                data_charge_c = cmap1(data_charge_c)

                qdata = self.charges(sol_phi_bar)

                imgd = Image.fromarray((data_charge_c[:, :, :3] * 255).astype(np.uint8))
                imgd = ImageOps.flip(imgd)
                Idraw = ImageDraw.Draw(imgd)
                Idraw.text((20,20),
                            "Q+: %.4f\nQ-: %.4f\nQtotal: %.4f"%qdata,
                            fill=(0,0,0))
                if fromloaded.any():
                    Idraw.text((self.ntot*factor-75,20),
                                "time: %.2f"%load_tsteps[i],
                                fill=(0,0,0))
                else:
                    Idraw.text((self.ntot*factor-75,20),
                                "time: %.2f"%self.timesteps[i],
                                fill=(0,0,0))
                videod.write(cv2.cvtColor(np.array(imgd), cv2.COLOR_RGB2BGR))   

            if (i%20==0):
                # print('average absolute charge: ',np.average(np.abs(data_charge)))
                print(i)

        try: videoa.release() 
        except: pass
        try: videob.release() 
        except: pass
        try: videoc.release() 
        except: pass
        try: videod.release() 
        except: pass

        return
    
    def charges(self,phi_bar,printit=False):
        phi_bar_p = np.zeros((2,self.ntot,self.ntot)).astype(complex)
        phi_bar_n = np.zeros((2,self.ntot,self.ntot)).astype(complex)
        phi_bar_p[0,...] = phi_bar[0,...]
        phi_bar_n[1,...] = phi_bar[1,...]

        phi = self.phi_bar_to_phi(phi_bar_p)
        varphi,idtvarphi = self.phi_to_psi(phi)
        rho = self.psi_to_rho(varphi,idtvarphi)
        Q_pos = np.sum(rho)*(self.l/self.n2)**2

        phi = self.phi_bar_to_phi(phi_bar_n)
        varphi,idtvarphi = self.phi_to_psi(phi)
        rho = self.psi_to_rho(varphi,idtvarphi)
        Q_neg = np.sum(rho)*(self.l/self.n2)**2

        if printit:
            print('Total positive charge: %.4f e'%Q_pos)
            print('Total negative charge: %.4f e'%Q_neg)
            print('Total charge: %.4f e'%(Q_pos+Q_neg))

        return (Q_pos,Q_neg,Q_pos+Q_neg)
    