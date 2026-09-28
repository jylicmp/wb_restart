import numpy as np
import wannierberri as wb
from wannierberri.system.system_R import System_R
from wannierberri.fourier.rvectors import Rvectors
from wannierberri.calculators.static import DOS, CumDOS


def model():
    s = System_R(silent=True)
    s.set_real_lattice(np.eye(3)*3)
    s.num_wann = 2
    s.wannier_centers_cart = np.zeros((2, 3))
    s.rvec = Rvectors(lattice=s.real_lattice, iRvec=np.array([[0,0,0],[1,0,0],[-1,0,0]]),
                     shifts_left_red=s.wannier_centers_red)
    h = np.array([[[.15,.2],[.2,-.3]], [[.25,.1],[.04,-.15]], [[.25,.04],[.1,-.15]]], dtype=complex)
    s.set_R_mat('Ham', h)
    s.set_pointgroup()
    return s


def setup():
    system = model()
    grid = wb.Grid(system, NKdiv=[2,2,1], NKFFT=[2,2,2])
    calcs = {'dos': DOS(Efermi=np.linspace(-1,1,31)), 'cum': CumDOS(Efermi=np.linspace(-1,1,31))}
    return system, grid, calcs


