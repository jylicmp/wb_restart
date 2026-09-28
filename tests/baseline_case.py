"""Small scientific model runnable with both untouched and restart environments."""
import json
from pathlib import Path
import sys
import numpy as np
import wannierberri as wb
from model_fixture import setup

system,grid,calcs=setup()
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
result=wb.run(system,grid,calcs,parallel=False,adpt_num_iter=2,fout_name=str(root/'result'))
np.savez(root/'tensors.npz',**{k:v.data for k,v in result.results.items()})
(root/'version.json').write_text(json.dumps({'version':wb.__version__,'module':wb.__file__}))
