#!/usr/bin/env python3
"""Verify completed iterations and retained partial outputs after the live-copy run."""
import argparse
import json
from pathlib import Path
import numpy as np
from wannierberri.restart import sha256_file

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root;control=root/'_restart_test';manifest=json.loads((root/'_tmp_wb'/'restart.json').read_text())
for iteration in ('0','1'):
    entry=manifest['iterations'][iteration]
    assert entry['phase']=='complete'
    assert sha256_file(root/'_tmp_wb'/entry['metadata'])==entry['sha256']
latest=json.loads((root/'partial'/'latest.json').read_text())
directory=root/'partial'/latest['snapshot'];meta=json.loads((directory/'metadata.json').read_text())
assert meta['complete'] and meta['completed_points']==meta['total_points']==620775
assert not meta['renormalized'];assert np.isclose(meta['completed_weight'],1.,rtol=0,atol=1e-9)
for name,digest in meta['files'].items():assert sha256_file(directory/name)==digest
errors={}
with np.load(directory/'tensors.npz',allow_pickle=False) as data:
    np.testing.assert_array_equal(data['completed_indices'],np.arange(620775))
    for key,entry in meta['calculators'].items():
        with np.load(root/'restart-results'/f'CrSe_Qorb_GaoXiao-{key}_iter-0000.npz',allow_pickle=False) as official:
            actual=data[entry['prefix']+'_data'];expected=official['data']
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)
            errors[key]=float(np.max(np.abs(actual-expected)))
        with np.load(root/'restart-results'/f'CrSe_Qorb_GaoXiao-{key}_iter-0001.npz',allow_pickle=False) as refined:
            assert np.all(np.isfinite(refined['data']))
snapshots=[json.loads(p.read_text()) for p in (root/'partial').glob('snapshot-*/metadata.json')]
counts=sorted(set(m['completed_points'] for m in snapshots))
assert counts[-1]==620775
report={'status':'passed','completed_iterations':[0,1],'partial_snapshots':len(snapshots),
        'partial_completed_counts':counts,'final_partial_vs_official_max_abs':errors,
        'configuration':manifest['fingerprint'],'dynamic_source_copy':True}
(control/'acceptance.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
