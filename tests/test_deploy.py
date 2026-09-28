"""Deployment safety checks in a disposable fake environment, never a live env."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DeployTests(unittest.TestCase):
    def test_isolation_and_obsolete_managed_file_removal(self):
        root=Path(__file__).resolve().parents[1]
        script=root/'scripts/deploy.py'
        with tempfile.TemporaryDirectory() as directory:
            temp=Path(directory)
            wrong=temp/'wberri_v1.8'
            p=subprocess.run([sys.executable,str(script),'--environment',str(wrong),'--allow-dirty'],
                             capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertFalse(wrong.exists())
            env=temp/'wberri_v1.8_restart'
            package=env/'lib/python3.12/site-packages/wannierberri'
            package.mkdir(parents=True)
            stale=package/'old_deployment.py';stale.write_text('stale')
            keep=package/'unmanaged.txt';keep.write_text('preserve')
            (package/'.wb_restart_deployment.json').write_text(json.dumps({'sha256':{'old_deployment.py':'old'}}))
            p=subprocess.run([sys.executable,str(script),'--environment',str(env),'--allow-dirty'],
                             capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertFalse(stale.exists())
            self.assertEqual(keep.read_text(),'preserve')
            source=root/'wannierberri/run.py';target=package/'run.py'
            self.assertEqual(source.read_bytes(),target.read_bytes())
            self.assertNotEqual((source.stat().st_dev,source.stat().st_ino),
                                (target.stat().st_dev,target.stat().st_ino))


if __name__=='__main__':unittest.main(verbosity=2)
