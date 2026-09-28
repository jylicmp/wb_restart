#!/usr/bin/env python3
"""Inspect a trusted checkpoint, or resume to a final global iteration."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import wannierberri as wb
from wannierberri.restart import Checkpoint, configuration, choose_iteration


def load_config(path):
    spec=importlib.util.spec_from_file_location('wb_user_config',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg=module.build()
    cfg.setdefault('parameters_K',{})
    cfg.setdefault('use_irred_kpt',True)
    cfg.setdefault('symmetrize',True)
    if cfg['use_irred_kpt']: cfg['symmetrize']=True
    cfg.setdefault('adpt_mesh',2)
    cfg.setdefault('adpt_fac',1)
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True,help='Python file defining build() -> run kwargs')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--inspect',action='store_true',help='Read only; no checkpoint or output writes')
    p.add_argument('--recover',action='store_true',help='Allow reconstruction of legacy iteration 0')
    p.add_argument('--on-corrupt',choices=['error','recompute'],default='error')
    p.add_argument('--iteration',type=int,default=-1)
    p.add_argument('--until-iteration',type=int,default=20)
    p.add_argument('--output',default='CrSe_Qorb_GaoXiao')
    p.add_argument('--serial',action='store_true')
    args=p.parse_args()
    cfg=load_config(args.config)
    manifest=args.checkpoint/'restart.json'
    if manifest.exists():
        indices=json.loads(manifest.read_text())['iterations']
    else:
        indices=[int(f.stem.split('-')[-1]) for f in args.checkpoint.glob('factors_iter-*.npy')]
    start=choose_iteration(indices,args.iteration)
    if args.inspect:
        mesh=cfg['adpt_mesh']
        if mesh is not None and np.max(mesh)>1:
            mesh=np.array([mesh]*3 if np.isscalar(mesh) else mesh)
        config=configuration(cfg['system'],cfg['grid'],cfg['calculators'],cfg['parameters_K'],
                             cfg['use_irred_kpt'],cfg['symmetrize'],mesh,cfg['adpt_fac'])
        store=Checkpoint(args.checkpoint,config,cfg['calculators'])
        iteration,points,factors=store.load(cfg['grid'],cfg['use_irred_kpt'],args.iteration,args.recover)
        _,report=store.restore(points,readonly=True)
        report.update(iteration=iteration,points=len(points),configuration=store.fingerprint,
                      legacy_provenance=store.manifest.get('legacy_provenance'))
        print(json.dumps(report,indent=2))
        return
    if args.until_iteration<start:
        p.error('Final iteration precedes selected checkpoint')
    if not args.serial:
        wb.ray_init()
    wb.run(**cfg,restart=True,restart_recover=args.recover,restart_on_corrupt=args.on_corrupt,
           restart_iteration=args.iteration,adpt_num_iter=args.until_iteration-start,
           allow_restart=True,dump_results=True,file_Klist_path=str(args.checkpoint),
           fout_name=args.output,parallel=not args.serial)


if __name__=='__main__': main()
