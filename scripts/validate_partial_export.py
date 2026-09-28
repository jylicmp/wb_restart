#!/usr/bin/env python3
"""Validate the CLI using an existing independent legacy sample checkpoint."""
import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys
import numpy as np
from wannierberri.restart import sha256_file

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--work',type=Path,required=True)
p.add_argument('--config',type=Path,required=True)
a=p.parse_args()
a.work.mkdir(parents=True,exist_ok=False)
before={f.name:sha256_file(f) for f in a.source.iterdir() if f.is_file()}
command=[sys.executable,'-B',str(Path(__file__).with_name('restart_run.py')),
         '--config',str(a.config),'--checkpoint',str(a.source),'--recover',
         '--export-partial','--partial-output',str(a.work/'partial')]
with (a.work/'export.log').open('w') as log:
    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=a.work)
assert before=={f.name:sha256_file(f) for f in a.source.iterdir() if f.is_file()}
weights=np.load(a.source/'factors_iter-00000000.npy',allow_pickle=False)
reference=None;indices=[]
for f in sorted(a.source.glob('_Kp-*.pickle'),key=lambda f:int(f.stem.split('-')[-1])):
    i=int(f.stem.split('-')[-1]);indices.append(i)
    with f.open('rb') as stream:part=pickle.load(stream)*weights[i]
    reference=part if reference is None else reference+part
pointer=json.loads((a.work/'partial'/'latest.json').read_text())
directory=a.work/'partial'/pointer['snapshot']
meta=json.loads((directory/'metadata.json').read_text())
assert meta['completed_points']==len(indices)==8
assert meta['total_points']==len(weights)==620775
assert not meta['complete'] and not meta['renormalized']
np.testing.assert_allclose(meta['completed_weight'],weights[indices].sum(),rtol=1e-14,atol=0)
errors={}
with np.load(directory/'tensors.npz',allow_pickle=False) as data:
    np.testing.assert_array_equal(data['completed_indices'],indices)
    for key,entry in meta['calculators'].items():
        for field in ('data','dataSmooth'):
            actual=data[entry['prefix']+'_'+field];expected=getattr(reference.results[key],field)
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)
            errors[key+'.'+field]=float(np.max(np.abs(actual-expected)))
report={'status':'passed','read_only':True,'sample_count':len(indices),'errors':errors,'rtol':1e-10,'atol':1e-12}
(a.work/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
