#!/usr/bin/env python3
"""Read original weights, copy stable samples, and recompute at most eight points.
Run only inside the approved small Slurm allocation. Never writes to source data.
"""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import time
import numpy as np
import ray
import wannierberri as wb
from wannierberri.data_K import get_data_k
from wannierberri.result import ResultDict
from wannierberri.restart import sha256_file,validate_result,atomic_write


def deployment():
    package=Path(wb.__file__).parent
    record=json.loads((package/'.wb_restart_deployment.json').read_text())
    assert not record['dirty'], 'Acceptance requires a committed deployment'
    assert all(sha256_file(package/name)==checksum for name,checksum in record['sha256'].items())
    return {'module':str(package),'commit':record['commit'],'sha256':record['sha256']}


def stable_copy(source,destination):
    before=source.stat()
    shutil.copyfile(source,destination)
    after=source.stat()
    assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
    assert sha256_file(source)==sha256_file(destination)
    return sha256_file(destination)


@ray.remote
def evaluate(kp,system,grid,calculators):
    identity=deployment()
    with get_data_k(system,kp.Kp_fullBZ,grid=grid,Kpoint=kp) as data:
        result=ResultDict({k:c(data) for k,c in calculators.items()})
    return system.pointgroup.symmetrize(result),identity


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--original-package',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--cpus',type=int,default=4)
    args=p.parse_args()
    if args.work.resolve()==args.source.resolve() or args.source.resolve() in args.work.resolve().parents:
        p.error('Use a separate validation directory')
    args.work.mkdir(parents=True,exist_ok=False)
    started=time.time()
    identity=deployment()
    original=json.loads(args.baseline.read_text())['sha256']
    assert all(sha256_file(args.original_package/name)==checksum for name,checksum in original.items())
    report={'status':'running','dynamic_snapshot':True,'deployment':identity,'original_unchanged':True,'samples':[]}
    def save():
        atomic_write(args.work/'validation.json',lambda f:f.write(json.dumps(report,indent=2).encode()))
    save()
    spec=importlib.util.spec_from_file_location('crse_config',args.config)
    config=importlib.util.module_from_spec(spec);spec.loader.exec_module(config)
    cfg=config.build()
    grid=cfg['grid'];system=cfg['system'];calcs=cfg['calculators']
    assert len(system.pointgroup)==6
    np.testing.assert_array_equal(grid.div,[155]*3)
    np.testing.assert_array_equal(grid.FFT,[11]*3)
    weight_path=args.source/'factors_iter-00000000.npy'
    report['weights_sha256']=stable_copy(weight_path,args.work/weight_path.name)
    expected=np.load(args.work/weight_path.name,allow_pickle=False)
    points=grid.get_K_list(use_symmetry=True)
    factors=np.array([kp.factor for kp in points])
    assert len(points)==620775
    np.testing.assert_array_equal(expected,factors)
    assert np.isclose(factors.sum(),1.,rtol=0,atol=1e-12)
    available={int(p.name[4:-7]) for p in args.source.iterdir()
               if p.name.startswith('_Kp-') and p.name.endswith('.pickle') and p.name[4:-7].isdigit()}
    assert all(0<=i<len(points) for i in available)
    indices=[]
    # Cover every symmetry-weight class, then spread across the index range.
    for weight in np.unique(factors):
        index=next((int(i) for i in np.flatnonzero(factors==weight) if int(i) in available),None)
        if index is not None and index not in indices:indices.append(index)
    for frac in (0,.25,.5,.75,1):
        target=round((len(points)-1)*frac)
        index=min(available,key=lambda i:abs(i-target))
        if index not in indices:indices.append(index)
    indices=indices[:8]
    report.update(points=len(points),available_files=len(available),missing_files=len(points)-len(available),
                  weights_equal=True,weight_sum=float(factors.sum()),sample_indices=indices)
    references={}
    selected={i:points[i] for i in indices}
    for i in indices:
        dest=args.work/f'_Kp-{i}.pickle'
        checksum=stable_copy(args.source/dest.name,dest)
        with dest.open('rb') as f: references[i]=pickle.load(f)
        validate_result(references[i],calcs)
        report['samples'].append({'index':i,'K':selected[i].Kp_fullBZ.tolist(),
                                  'factor':float(factors[i]),'file_sha256':checksum})
    del points;gc.collect()
    save()
    ray.init(num_cpus=args.cpus,include_dashboard=False,
             _temp_dir=str(args.work/'ray'))
    try:
        rs,rg,rc=ray.put(system),ray.put(grid),ray.put(calcs)
        tasks={evaluate.remote(selected[i],rs,rg,rc):i for i in indices}
        while tasks:
            ready,_=ray.wait(list(tasks),num_returns=1,timeout=60)
            for ref in ready:
                i=tasks.pop(ref)
                result,worker=ray.get(ref)
                assert worker==identity,'Worker uses a different code deployment'
                sample=next(s for s in report['samples'] if s['index']==i)
                sample['errors']={}
                for key in calcs:
                    actual=result.results[key]; expected_result=references[i].results[key]
                    for field in ('data','dataSmooth'):
                        a,b=getattr(actual,field),getattr(expected_result,field)
                        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
                        sample['errors'][key+'.'+field]=float(np.max(np.abs(a-b)))
                sample['passed']=True
                save()
    finally:
        ray.shutdown()
    assert all(sha256_file(args.original_package/name)==checksum for name,checksum in original.items())
    report.update(status='passed',elapsed_seconds=time.time()-started,
                  original_unchanged_after=True,rtol=1e-10,atol=1e-12)
    save()
    print(json.dumps({k:v for k,v in report.items() if k!='deployment'},indent=2))


if __name__=='__main__':main()
