"""Run with unittest in the cloned environment; no optional pytest dependency."""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import wannierberri as wb
from wannierberri.system.system_R import System_R
from wannierberri.fourier.rvectors import Rvectors
from wannierberri.calculators.static import DOS, CumDOS
from wannierberri.restart import (Checkpoint, configuration, IncompatibleCheckpoint,
                                 CorruptCheckpoint, RestartError, choose_iteration)

runmod = importlib.import_module('wannierberri.run')


def model():
    s = System_R(silent=True)
    s.set_real_lattice(np.eye(3)*3)
    s.num_wann = 2
    s.wannier_centers_cart = np.zeros((2, 3))
    s.rvec = Rvectors(lattice=s.real_lattice, iRvec=np.array([[0,0,0],[1,0,0],[-1,0,0]]),
                     shifts_left_red=s.wannier_centers_red)
    h = np.array([[[.15,.2],[.2,-.3]], [[.25,.1],[.04,-.15]], [[.25,.04],[.1,-.15]]], dtype=complex)
    s.set_R_mat('Ham', h)
    s.set_pointgroup()
    return s


def setup():
    system = model()
    grid = wb.Grid(system, NKdiv=[2,2,1], NKFFT=[2,2,2])
    calcs = {'dos': DOS(Efermi=np.linspace(-1,1,31)), 'cum': CumDOS(Efermi=np.linspace(-1,1,31))}
    return system, grid, calcs


class RestartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.system, self.grid, self.calcs = setup()
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()

    def tearDown(self):
        self.redirect.__exit__(None, None, None)
        self.tmp.cleanup()

    def run_calc(self, name, iterations=0, **kwargs):
        return wb.run(self.system, self.grid, self.calcs, parallel=False, allow_restart=True,
                      dump_results=True, file_Klist_path=str(self.root/name),
                      fout_name=str(self.root/(name+'-result')), adpt_num_iter=iterations, **kwargs)

    def equal(self, a, b):
        for k in a.results:
            np.testing.assert_allclose(a.results[k].data, b.results[k].data, rtol=1e-10, atol=1e-12)

    def checkpoint(self, path):
        cfg = configuration(self.system,self.grid,self.calcs,{},True,True,np.array([2,2,2]),1)
        return Checkpoint(self.root/path,cfg,self.calcs)

    def metadata(self, name, iteration):
        m = json.loads((self.root/name/'restart.json').read_text())
        with (self.root/name/m['iterations'][str(iteration)]['metadata']).open('rb') as f:
            return pickle.load(f)

    def test_continuous_and_restart_complete(self):
        ref = self.run_calc('full', 2)
        self.run_calc('split',1)
        got = self.run_calc('split',1,restart=True)
        self.equal(ref,got)
        self.equal(got,self.run_calc('split',0,restart=True))
        p1,f1=self.metadata('full',2); p2,f2=self.metadata('split',2)
        np.testing.assert_array_equal(f1,f2)
        np.testing.assert_array_equal([p.K for p in p1],[p.K for p in p2])

    def test_interrupted_first_and_later_round(self):
        ref=self.run_calc('full',2)
        original=runmod.get_data_k
        for stop in (2,6):
            count=[0]
            def failing(*args,**kwargs):
                count[0]+=1
                if count[0]==stop: raise RuntimeError('simulated interruption')
                return original(*args,**kwargs)
            name='stop'+str(stop)
            with patch.object(runmod,'get_data_k',side_effect=failing):
                with self.assertRaisesRegex(RuntimeError,'simulated'): self.run_calc(name,2)
            m=json.loads((self.root/name/'restart.json').read_text())
            start=max(map(int,m['iterations']))
            got=self.run_calc(name,2-start,restart=True)
            self.equal(ref,got)
            p1,f1=self.metadata('full',2); p2,f2=self.metadata(name,2)
            np.testing.assert_array_equal(f1,f2)
            np.testing.assert_array_equal([p.K for p in p1],[p.K for p in p2])

    def test_legacy_recovery_and_readonly(self):
        ref=self.run_calc('full')
        store=self.checkpoint('full')
        _,points,factors=store.load(self.grid,True)
        dest=self.root/'legacy'; dest.mkdir()
        from wannierberri.restart import load_point
        for i,kp in enumerate(points):
            if i==1: continue
            with (dest/f'_Kp-{i}.pickle').open('wb') as f: pickle.dump(load_point(kp),f)
        np.save(dest/'factors_iter-00000000.npy',factors)
        before={p.name:p.read_bytes() for p in dest.iterdir()}
        inspect_store=self.checkpoint('legacy')
        _,ps,_=inspect_store.load(self.grid,True,recover=True)
        _,report=inspect_store.restore(ps,readonly=True)
        self.assertEqual(report['missing'],[1])
        self.assertEqual(before,{p.name:p.read_bytes() for p in dest.iterdir()})
        with self.assertRaises(RestartError): self.run_calc('legacy',restart=True)
        self.equal(ref,self.run_calc('legacy',restart=True,restart_recover=True))
        self.equal(ref,self.run_calc('legacy',restart=True))

    def test_missing_corrupt_moved(self):
        ref=self.run_calc('full')
        shutil.copytree(self.root/'full', self.root/'moved')
        files=sorted((self.root/'moved').glob('_Kp-*.pickle'))
        files[0].unlink(); files[1].write_bytes(b'\x80\x05')
        with self.assertRaises(CorruptCheckpoint): self.run_calc('moved',restart=True)
        self.equal(ref,self.run_calc('moved',restart=True,restart_on_corrupt='recompute'))
        self.assertEqual(len(list((self.root/'moved').glob('*.corrupt-*'))),1)

    def test_configuration_and_identity_fail_closed(self):
        self.run_calc('full')
        with self.assertRaises(RestartError): self.run_calc('full')
        self.calcs['dos'].Efermi=self.calcs['dos'].Efermi+0.01
        with self.assertRaises(IncompatibleCheckpoint): self.run_calc('full',restart=True,restart_on_corrupt='recompute')
        self.calcs['dos'].Efermi=self.calcs['dos'].Efermi-0.01
        # Fresh setup avoids floating point roundtrip effects in config hash.
        self.system,self.grid,self.calcs=setup()
        files=sorted((self.root/'full').glob('_Kp-*.pickle'))
        files[0].write_bytes(files[1].read_bytes())
        with self.assertRaises(IncompatibleCheckpoint): self.run_calc('full',restart=True,restart_on_corrupt='recompute')

    def test_historical_iteration_and_weights_sort(self):
        ref=self.run_calc('full',2)
        self.equal(ref,self.run_calc('full',2,restart=True,restart_iteration=0))
        self.assertEqual(choose_iteration([10,0,2],-1),10)
        self.assertEqual(choose_iteration([10,0,2],-2),2)
        with self.assertRaises(RestartError): choose_iteration([0,2],1)

    def test_checkpoint_commit_interruption(self):
        import wannierberri.restart as restart
        ref=self.run_calc('full',1)
        self.run_calc('split')
        old=(self.root/'split'/'restart.json').read_bytes()
        original=restart.atomic_write
        def fail_manifest(path,writer):
            if Path(path).name=='restart.json': raise RuntimeError('manifest boundary')
            return original(path,writer)
        with patch.object(restart,'atomic_write',side_effect=fail_manifest):
            with self.assertRaisesRegex(RuntimeError,'boundary'): self.run_calc('split',1,restart=True)
        self.assertEqual(old,(self.root/'split'/'restart.json').read_bytes())
        self.equal(ref,self.run_calc('split',1,restart=True))

    def test_result_atomic_interruption_and_lock(self):
        import wannierberri.restart as restart
        ref=self.run_calc('full')
        original=restart.os.replace
        def fail_result(src,dst):
            if Path(dst).name.startswith('_Kp-'): raise RuntimeError('result boundary')
            return original(src,dst)
        with patch.object(restart.os,'replace',side_effect=fail_result):
            with self.assertRaisesRegex(RuntimeError,'boundary'): self.run_calc('split')
        self.equal(ref,self.run_calc('split',restart=True))
        import fcntl
        with open(self.root/'split'/'.restart.lock','a+b') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(RestartError): self.run_calc('split',restart=True)


if __name__=='__main__': unittest.main(verbosity=2)
