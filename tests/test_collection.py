import contextlib
import importlib
import io
import sys
import time
import tracemalloc
import types
import unittest
from unittest.mock import patch

runmod=importlib.import_module('wannierberri.run')


class Point:
    def __init__(self,i):
        self.i=i
        self.was_evaluated_flag=False
        self.result=None
        self.writes=0
    def set_result(self,value):
        self.result=value
        self.was_evaluated_flag=True
    def get_result_factor(self): return self.result
    def dump_result(self):
        self.writes+=1
        self.result=None
    def clear_result(self): self.result=None


class CollectionTests(unittest.TestCase):
    def test_bounded_once_only_collection(self):
        n=50000
        points=[Point(i) for i in range(n)]
        # Pre-existing results must not be dispatched or counted again.
        points[0].was_evaluated_flag=True
        objects={}; calls=[]; peak=[0]
        def remote(kp,**kwargs):
            ref=kp.i
            objects[ref]=float(kp.i)
            peak[0]=max(peak[0],len(objects))
            return ref
        def wait(refs,num_returns,timeout):
            ready=refs[-num_returns:]  # completion order differs from submission
            return ready,refs[:-num_returns]
        def get(refs):
            calls.extend(refs)
            return [objects.pop(r) for r in refs]
        fake=types.SimpleNamespace(wait=wait,get=get)
        t=time.perf_counter();tracemalloc.start()
        with patch.dict(sys.modules,{'ray':fake}),patch.object(runmod,'get_ray_cpus_count',return_value=32):
            with contextlib.redirect_stdout(io.StringIO()):
                count,result=runmod.process(types.SimpleNamespace(remote=remote),points,True,True,{},True)
        _,memory=tracemalloc.get_traced_memory();tracemalloc.stop()
        self.assertEqual(count,n-1)
        self.assertEqual(result,n*(n-1)/2)
        self.assertEqual(len(calls),len(set(calls)))
        self.assertEqual(set(calls),set(range(1,n)))
        self.assertLessEqual(peak[0],256)
        self.assertTrue(all(p.writes==1 for p in points[1:]))
        self.assertEqual(points[0].writes,0)
        print({'tasks':n-1,'peak_pending':peak[0],'seconds':time.perf_counter()-t,'collection_peak_bytes':memory})


if __name__=='__main__':unittest.main(verbosity=2)
