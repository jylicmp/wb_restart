#!/usr/bin/env python3
"""Wait for the allocated Ray nodes and verify every node's installed package."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import wannierberri as wb


def deployment():
    path=Path(wb.__file__).parent
    record=json.loads((path/'.wb_restart_deployment.json').read_text())
    assert not record['dirty']
    assert all(hashlib.sha256((path/name).read_bytes()).hexdigest()==digest for name,digest in record['sha256'].items())
    return str(path),record


@ray.remote(num_cpus=0)
def identity():
    return {'host':socket.gethostname().split('.')[0],'python':sys.executable,'deployment':deployment()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--address',required=True);p.add_argument('--hosts',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();expected=deployment();ray.init(address=a.address)
    try:
        deadline=time.monotonic()+300
        while True:
            nodes=[n for n in ray.nodes() if n['Alive']]
            if len(nodes)==len(a.hosts):break
            if time.monotonic()>deadline:raise TimeoutError(f'Only {len(nodes)} Ray nodes joined')
            time.sleep(2)
        checks=ray.get([identity.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n['NodeID'],soft=False)).remote() for n in nodes],timeout=120)
        assert set(c['host'] for c in checks)==set(a.hosts),checks
        assert all(c['deployment']==expected for c in checks),'Mixed deployments'
        report={'status':'passed','address':a.address,'resources':ray.cluster_resources(),'workers':checks}
        a.output.write_text(json.dumps(report,indent=2))
        print(f'CLUSTER_VERIFIED: {len(checks)} hosts, {ray.cluster_resources().get("CPU")} Ray CPUs, commit {expected[1]["commit"]}',flush=True)
    finally:ray.shutdown()


if __name__=='__main__':main()
