"""Real Ray workers: numerical equivalence, missing-point recovery, deployment."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from wannierberri.partial import PartialWriter
import numpy as np
import ray
import wannierberri as wb
from model_fixture import setup

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
s,g,c=setup()
zero=wb.run(s,g,c,parallel=False,adpt_num_iter=0,fout_name=str(root/'zero'))
serial=wb.run(s,g,c,parallel=False,adpt_num_iter=2,fout_name=str(root/'serial'))
ray.init(num_cpus=4,include_dashboard=False,_temp_dir=tempfile.mkdtemp(prefix='wbr-'),object_store_memory=512*1024**2)
try:
    @ray.remote
    def worker_version():
        import wannierberri as wb
        p=Path(wb.__file__).parent
        return str(p),json.loads((p/'.wb_restart_deployment.json').read_text())
    workers=ray.get([worker_version.remote() for _ in range(4)])
    assert all(w==workers[0] for w in workers)
    assert workers[0][0]==str(Path(wb.__file__).parent)
    assert workers[0][1]==json.loads((Path(wb.__file__).parent/'.wb_restart_deployment.json').read_text())
    def run(restart=False,iterations=2):
        return wb.run(s,g,c,parallel=True,adpt_num_iter=iterations,
                      allow_restart=True,dump_results=True,restart=restart,
                      file_Klist_path=str(root/'checkpoint'),fout_name=str(root/'parallel'),
                      partial_save_interval=900,partial_output_dir=root/'partial')
    parallel=run()
    # A real completed result is removed from the independent test checkpoint.
    next((root/'checkpoint').glob('_Kp-*.pickle')).unlink()
    recovered=run(True,0)
    errors={}
    for key in c:
        for name,result in [('parallel',parallel),('recovered',recovered)]:
            np.testing.assert_allclose(result.results[key].data,serial.results[key].data,rtol=1e-10,atol=1e-12)
            errors[key+'.'+name]=float(np.max(np.abs(result.results[key].data-serial.results[key].data)))
    # Interrupt the driver after two real Ray results have been saved.
    observe=PartialWriter.observe
    def interrupted(self,point,subtotal,base=None):
        observe(self,point,subtotal,base)
        if self.count==2:raise RuntimeError('intentional driver interruption')
    kwargs=dict(parallel=True,allow_restart=True,dump_results=True,
                file_Klist_path=str(root/'interrupted'),fout_name=str(root/'interrupted-result'),
                partial_save_interval=900,partial_output_dir=root/'interrupted-partial')
    with patch.object(PartialWriter,'observe',interrupted):
        try:wb.run(s,g,c,**kwargs)
        except RuntimeError as exc:assert 'intentional' in str(exc)
        else:raise AssertionError('Expected interruption')
    recovered_zero=wb.run(s,g,c,restart=True,**kwargs)
    for key in c:
        np.testing.assert_allclose(recovered_zero.results[key].data,zero.results[key].data,rtol=1e-10,atol=1e-12)
        errors[key+'.iteration_zero_recovered']=float(np.max(np.abs(recovered_zero.results[key].data-zero.results[key].data)))
    snapshots=[json.loads(p.read_text()) for p in (root/'interrupted-partial').glob('snapshot-*/metadata.json')]
    assert sorted(m['completed_points'] for m in snapshots)==[1,2,4]
    latest=json.loads((root/'interrupted-partial'/'latest.json').read_text())
    directory=root/'interrupted-partial'/latest['snapshot']
    meta=json.loads((directory/'metadata.json').read_text())
    with np.load(directory/'tensors.npz',allow_pickle=False) as arrays:
        for key,entry in meta['calculators'].items():
            for field in ('data','dataSmooth'):
                np.testing.assert_allclose(arrays[entry['prefix']+'_'+field],getattr(zero.results[key],field),rtol=1e-10,atol=1e-12)
    (root/'ray-validation.json').write_text(json.dumps({'status':'passed','errors':errors,'commit':workers[0][1]['commit'],
                                                       'worker_module':workers[0][0]},indent=2))
finally:
    ray.shutdown()
