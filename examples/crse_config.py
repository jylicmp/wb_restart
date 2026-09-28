"""Original CrSe MQM parameters; set WB_TB_FILE to the local model path."""
import os
import numpy as np
import wannierberri as wb
from wannierberri import calculators as calc
from wannierberri.smoother import FermiDiracSmoother


def build():
    system = wb.System_tb(tb_file=os.environ['WB_TB_FILE'], qorb=True)
    system.set_pointgroup(['My*TimeReversal', 'C3z'])
    energies = np.linspace(6.7412, 7.7412, 201, True)
    options = dict(Efermi=energies, smoother=FermiDiracSmoother(energies, 1))
    return dict(system=system, grid=wb.Grid(system, NK=1700, NKFFT=11),
                calculators={'Qorb_IMD': calc.static.Qorb_GaoXiao_IMD(**options),
                             'Qorb_QMD': calc.static.Qorb_GaoXiao_QMD(**options)},
                use_irred_kpt=True, symmetrize=True, adpt_mesh=2, adpt_fac=1,
                parameters_K={})
