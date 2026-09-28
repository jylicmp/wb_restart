"""Real Ray workers: numerical equivalence, missing-point recovery, deployment."""
import json
from pathlib import Path
import sys
import numpy as np
import ray
import wannierberri as wb
from model_fixture import setup

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
s,g,c=setup()
serial=wb.run(s,g,c,parallel=False,adpt_num_iter=2,fout_name=str(root/'serial'))
ray.init(num_cpus=4,include_dashboard=False,_temp_dir=str(root/'ray'))
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
                      file_Klist_path=str(root/'checkpoint'),fout_name=str(root/'parallel'))
    parallel=run()
    # A real completed result is removed from the independent test checkpoint.
    next((root/'checkpoint').glob('_Kp-*.pickle')).unlink()
    recovered=run(True,0)
    errors={}
    for key in c:
        for name,result in [('parallel',parallel),('recovered',recovered)]:
            np.testing.assert_allclose(result.results[key].data,serial.results[key].data,rtol=1e-10,atol=1e-12)
            errors[key+'.'+name]=float(np.max(np.abs(result.results[key].data-serial.results[key].data)))
    (root/'ray-validation.json').write_text(json.dumps({'status':'passed','errors':errors,'commit':workers[0][1]['commit'],
                                                       'worker_module':workers[0][0]},indent=2))
finally:
    ray.shutdown()
