"""Observation snapshots must never change accumulation or restart authority."""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import wannierberri as wb
from wannierberri.restart import Checkpoint, configuration, load_point, IncompatibleCheckpoint, RestartError
from wannierberri.partial import PartialWriter, validate_output
from model_fixture import setup

runmod=importlib.import_module('wannierberri.run')
partialmod=importlib.import_module('wannierberri.partial')


class PartialTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.s,self.g,self.c=setup()
        self.redirect=contextlib.redirect_stdout(io.StringIO());self.redirect.__enter__()

    def tearDown(self):
        self.redirect.__exit__(None,None,None);self.tmp.cleanup()

    def run_calc(self,name,**kwargs):
        return wb.run(self.s,self.g,self.c,parallel=False,allow_restart=True,dump_results=True,
                      file_Klist_path=str(self.root/name),fout_name=str(self.root/(name+'-result')),**kwargs)

    def load(self,name):
        cfg=configuration(self.s,self.g,self.c,{},True,True,np.array([2]*3),1)
        store=Checkpoint(self.root/name,cfg,self.c)
        _,points,_=store.load(self.g,True)
        return store,points

    def snapshots(self,name):
        directory=self.root/name
        return [json.loads(p.read_text()) for p in sorted(directory.glob('snapshot-*/metadata.json'))]

    def latest(self,name):
        directory=self.root/name
        pointer=json.loads((directory/'latest.json').read_text())
        generation=directory/pointer['snapshot']
        return generation,json.loads((generation/'metadata.json').read_text())

    def compare(self,directory,result):
        meta=json.loads((directory/'metadata.json').read_text())
        with np.load(directory/'tensors.npz',allow_pickle=False) as data:
            for key,entry in meta['calculators'].items():
                for field in ('data','dataSmooth'):
                    np.testing.assert_allclose(data[entry['prefix']+'_'+field],getattr(result.results[key],field),rtol=1e-10,atol=1e-12)

    def test_first_final_and_every_snapshot_equals_selected_points(self):
        got=self.run_calc('run',partial_save_interval=900,partial_output_dir=self.root/'partial')
        snapshots=self.snapshots('partial')
        self.assertEqual(sorted(m['completed_points'] for m in snapshots),[1,4])
        store,points=self.load('run')
        for mpath in (self.root/'partial').glob('snapshot-*/metadata.json'):
            with np.load(mpath.parent/'tensors.npz',allow_pickle=False) as data:
                ids=data['completed_indices']
            expected=sum(load_point(points[i])*points[i].factor for i in ids)
            self.compare(mpath.parent,expected)
            m=json.loads(mpath.read_text());self.assertAlmostEqual(m['completed_weight'],sum(points[i].factor for i in ids))
        directory,meta=self.latest('partial');self.assertTrue(meta['complete']);self.compare(directory,got)
        self.assertFalse(meta['renormalized'])

    def test_resume_immediate_and_no_double_count(self):
        original=runmod.get_data_k;calls=[]
        def stop(*args,**kwargs):
            if len(calls)==2:raise RuntimeError('stop')
            calls.append(1);return original(*args,**kwargs)
        with patch.object(runmod,'get_data_k',side_effect=stop):
            with self.assertRaisesRegex(RuntimeError,'stop'):
                self.run_calc('run',partial_save_interval=900,partial_output_dir=self.root/'partial')
        got=self.run_calc('run',restart=True,partial_save_interval=900,partial_output_dir=self.root/'partial')
        self.assertEqual(sorted(m['completed_points'] for m in self.snapshots('partial')),[1,2,4])
        self.compare(self.latest('partial')[0],got)
        with patch.object(runmod,'get_data_k',side_effect=AssertionError('duplicate')):
            self.run_calc('run',restart=True,partial_save_interval=900,partial_output_dir=self.root/'partial')
        self.assertEqual(len(self.snapshots('partial')),4)

    def test_timer_no_change_and_failed_publication_retry(self):
        self.run_calc('full');store,points=self.load('full')
        for p in points:p.was_evaluated_flag=False
        with PartialWriter(self.root/'partial',store.fingerprint,points,900) as writer:
            total=None
            with patch.object(partialmod.time,'monotonic',return_value=0):
                p=points[0];total=load_point(p)*p.factor;writer.observe(p,total)
            with patch.object(partialmod.time,'monotonic',return_value=899):
                p=points[1];total+=load_point(p)*p.factor;writer.observe(p,total)
            self.assertEqual(len(self.snapshots('partial')),1)
            with patch.object(partialmod.time,'monotonic',return_value=900):
                writer.observe(p,total)
            self.assertEqual(len(self.snapshots('partial')),2)
            writer.publish(total);self.assertEqual(len(self.snapshots('partial')),2)
            old=(self.root/'partial'/'latest.json').read_bytes()
            atomic=partialmod.atomic_write
            def fail(path,func):
                if Path(path).name=='latest.json':raise OSError('disk failure')
                return atomic(path,func)
            p=points[2];total+=load_point(p)*p.factor
            with patch.object(partialmod.time,'monotonic',return_value=1800),patch.object(partialmod,'atomic_write',side_effect=fail):
                with self.assertWarnsRegex(RuntimeWarning,'continues'):writer.observe(p,total)
            self.assertEqual(old,(self.root/'partial'/'latest.json').read_bytes())
            with patch.object(partialmod.time,'monotonic',return_value=2700):writer.observe(p,total)
            self.assertEqual(self.latest('partial')[1]['completed_points'],3)

    def test_atomic_boundaries_keep_last_complete_snapshot(self):
        self.run_calc('full');store,points=self.load('full')
        with PartialWriter(self.root/'partial',store.fingerprint,points,0) as writer:
            first=load_point(points[0])*points[0].factor
            writer.observe(points[0],first)
            old=(self.root/'partial'/'latest.json').read_bytes()
            total=first+load_point(points[1])*points[1].factor
            atomic=partialmod.atomic_write
            for boundary in ('c0000.dat','tensors.npz','metadata.json','latest.json'):
                with self.subTest(boundary=boundary):
                    def fail(path,func):
                        if Path(path).name==boundary:raise OSError('interrupted')
                        return atomic(path,func)
                    with patch.object(partialmod,'atomic_write',side_effect=fail):
                        with self.assertWarns(RuntimeWarning):writer.observe(points[1],total)
                    self.assertEqual(old,(self.root/'partial'/'latest.json').read_bytes())
                    self.compare(self.latest('partial')[0],first)
            writer.publish(total)
            self.compare(self.latest('partial')[0],total)

    def test_output_failure_does_not_stop_calculation(self):
        with patch.object(PartialWriter,'_publish',side_effect=OSError('full disk')):
            with self.assertWarns(RuntimeWarning):got=self.run_calc('run',partial_save_interval=900,partial_output_dir=self.root/'partial')
        reference=self.run_calc('reference')
        for key in self.c:np.testing.assert_array_equal(got.results[key].data,reference.results[key].data)

    def test_readonly_export_sparse_corrupt_and_empty(self):
        self.run_calc('full');store,points=self.load('full')
        legacy=self.root/'legacy';legacy.mkdir()
        np.save(legacy/'factors_iter-00000000.npy',[p.factor for p in points])
        with (legacy/'_Kp-3.pickle').open('wb') as f:pickle.dump(load_point(points[3]),f)
        (legacy/'_Kp-1.pickle').write_bytes(b'\x80\x05')
        tests=Path(__file__).parent.resolve()
        cfg=self.root/'cfg.py';cfg.write_text('import sys\nsys.path.insert(0,'+repr(str(tests))+')\nfrom model_fixture import setup\ndef build():\n s,g,c=setup()\n return dict(system=s,grid=g,calculators=c)\n')
        before={p.name:p.read_bytes() for p in legacy.iterdir()}
        command=[sys.executable,'-B',str(tests.parent/'scripts/restart_run.py'),'--config',str(cfg),'--checkpoint',str(legacy),'--recover','--export-partial','--partial-output',str(self.root/'export')]
        result=subprocess.run(command,cwd=self.root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(before,{p.name:p.read_bytes() for p in legacy.iterdir()})
        directory,meta=self.latest('export');self.assertEqual(meta['completed_points'],1);self.assertEqual(meta['corrupt_points'],1);self.assertEqual(meta['missing_points'],2)
        self.compare(directory,load_point(points[3])*points[3].factor)
        (legacy/'_Kp-3.pickle').unlink()
        command[-1]=str(self.root/'empty')
        result=subprocess.run(command,cwd=self.root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        directory,meta=self.latest('empty');self.assertEqual(meta['completed_points'],0);self.assertFalse((directory/'tensors.npz').exists())

    def test_lock_configuration_paths_and_later_iteration(self):
        self.run_calc('full');store,points=self.load('full')
        with PartialWriter(self.root/'partial',store.fingerprint,points):
            with self.assertRaises(BlockingIOError):
                with PartialWriter(self.root/'partial',store.fingerprint,points):pass
        with self.assertRaises(IncompatibleCheckpoint):
            with PartialWriter(self.root/'partial','different',points):pass
        with self.assertRaises(RestartError):validate_output(self.root/'full'/'output',self.root/'full')
        (self.root/'alias').symlink_to(self.root/'full',target_is_directory=True)
        with self.assertRaises(RestartError):validate_output(self.root/'alias',self.root/'full')
        ref=self.run_calc('reference',adpt_num_iter=2)
        got=self.run_calc('enabled',adpt_num_iter=2,partial_save_interval=900,partial_output_dir=self.root/'enabled-partial')
        for key in self.c:np.testing.assert_array_equal(got.results[key].data,ref.results[key].data)
        self.assertTrue(all(m['iteration']==0 for m in self.snapshots('enabled-partial')))

    def test_existing_checkpoint_compatibility(self):
        source=os.environ.get('WB_PRE_PARTIAL_CHECKPOINT')
        if not source:self.skipTest('Old installed checkpoint supplied in cluster acceptance')
        shutil.copytree(source,self.root/'old')
        self.run_calc('old',restart=True,restart_iteration=0,partial_save_interval=900,partial_output_dir=self.root/'partial')
        self.assertTrue(self.latest('partial')[1]['complete'])

    def test_changing_file_is_excluded_from_readonly_sum(self):
        self.run_calc('full');store,points=self.load('full')
        restartmod=importlib.import_module('wannierberri.restart');original=restartmod.load_point
        def changing(point):
            result=original(point)
            if point.restart_index==1:
                with open(point.result_storage_path,'ab') as f:f.write(b'changed')
            return result
        with patch.object(restartmod,'load_point',side_effect=changing):total,report=store.restore(points,readonly=True)
        self.assertEqual(report['unstable'],[1]);self.assertEqual(report['valid'],3)


if __name__=='__main__':unittest.main()
