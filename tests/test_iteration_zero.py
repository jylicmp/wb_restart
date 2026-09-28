"""Iteration-0 recovery with sparse, unordered and repeatedly interrupted files."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import wannierberri as wb
from wannierberri.restart import Checkpoint, configuration, load_point
from model_fixture import setup

runmod = importlib.import_module('wannierberri.run')


class IterationZeroTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.redirect = contextlib.redirect_stdout(io.StringIO())
        self.redirect.__enter__()
        self.system, _, self.calcs = setup()
        self.grid = wb.Grid(self.system, NKdiv=[7, 4, 3], NKFFT=[2, 2, 2])
        self.reference = self.run_calc('reference')
        cfg = configuration(self.system, self.grid, self.calcs, {}, True, True,
                            np.array([2, 2, 2]), 1)
        store = Checkpoint(self.root/'reference', cfg, self.calcs)
        _, self.points, self.factors = store.load(self.grid, True)
        self.assertEqual(len(self.points), 84)
        self.coordinates = [tuple(k.Kp_fullBZ) for k in self.points]

    def tearDown(self):
        self.redirect.__exit__(None, None, None)
        self.tmp.cleanup()

    def run_calc(self, name, iterations=0, **kwargs):
        return wb.run(self.system, self.grid, self.calcs, parallel=False,
                      allow_restart=True, dump_results=True,
                      file_Klist_path=str(self.root/name),
                      fout_name=str(self.root/(name+'-result')),
                      adpt_num_iter=iterations, **kwargs)

    def equal(self, actual, expected=None):
        expected = self.reference if expected is None else expected
        for key in expected.results:
            for field in ('data', 'dataSmooth'):
                np.testing.assert_allclose(getattr(actual.results[key], field),
                                           getattr(expected.results[key], field),
                                           rtol=1e-10, atol=1e-12)

    def legacy(self, name, indices):
        dest = self.root/name
        dest.mkdir()
        np.save(dest/'factors_iter-00000000.npy', self.factors)
        for index in indices:
            with (dest/f'_Kp-{index}.pickle').open('wb') as stream:
                pickle.dump(load_point(self.points[index]), stream)
        return dest

    @contextlib.contextmanager
    def record_computation(self, calls, stop_after=None):
        original = runmod.get_data_k
        def tracked(*args, **kwargs):
            if stop_after is not None and len(calls) == stop_after:
                raise RuntimeError('iteration-zero interruption')
            calls.append(tuple(kwargs['Kpoint'].Kp_fullBZ))
            return original(*args, **kwargs)
        with patch.object(runmod, 'get_data_k', side_effect=tracked):
            yield

    def assert_only_missing(self, calls, saved):
        expected = [k for i, k in enumerate(self.coordinates) if i not in saved]
        self.assertCountEqual(calls, expected)
        self.assertEqual(len(calls), len(set(calls)), 'a point was computed twice')

    def test_legacy_completion_levels_and_unordered_holes(self):
        order = np.random.default_rng(174).permutation(84).tolist()
        cases = [[], [0], list(range(42)), order[:57], order[:83], order]
        for number, saved in enumerate(cases):
            with self.subTest(saved=len(saved)):
                name = f'legacy-{number}'
                dest = self.legacy(name, saved)
                # All cases deliberately lack K_list.pickle and restart.json.
                unchanged = {p.name: p.read_bytes() for p in dest.glob('_Kp-*')}
                calls = []
                with self.record_computation(calls):
                    self.equal(self.run_calc(name, restart=True, restart_recover=True))
                self.assert_only_missing(calls, set(saved))
                for filename, contents in unchanged.items():
                    self.assertEqual((dest/filename).read_bytes(), contents)
                with patch.object(runmod, 'get_data_k', side_effect=AssertionError('duplicate')):
                    self.equal(self.run_calc(name, restart=True))

    def test_native_interrupt_before_first_middle_and_last_point(self):
        for completed in (0, 1, 41, 83):
            with self.subTest(completed=completed):
                name = f'native-{completed}'
                calls = []
                with self.record_computation(calls, stop_after=completed):
                    with self.assertRaisesRegex(RuntimeError, 'iteration-zero'):
                        self.run_calc(name)
                manifest = json.loads((self.root/name/'restart.json').read_text())
                self.assertEqual(manifest['iterations']['0']['phase'], 'running')
                resumed = []
                with self.record_computation(resumed):
                    self.equal(self.run_calc(name, restart=True))
                self.assert_only_missing(calls+resumed, set())

    def test_sparse_legacy_recovery_interrupted_twice_then_refined(self):
        saved = set(np.random.default_rng(61).choice(84, 39, replace=False).tolist())
        self.legacy('repeated', sorted(saved, reverse=True))
        all_calls = []
        for attempt, limit in enumerate((7, 11)):
            calls = []
            with self.record_computation(calls, stop_after=limit):
                with self.assertRaisesRegex(RuntimeError, 'iteration-zero'):
                    self.run_calc('repeated', restart=True, restart_recover=attempt == 0)
            all_calls.extend(calls)
        calls = []
        with self.record_computation(calls):
            self.equal(self.run_calc('repeated', restart=True))
        self.assert_only_missing(all_calls+calls, saved)
        expected = self.run_calc('reference', 2, restart=True)
        actual = self.run_calc('repeated', 2, restart=True)
        self.equal(actual, expected)
        def metadata(name):
            m = json.loads((self.root/name/'restart.json').read_text())
            with (self.root/name/m['iterations']['2']['metadata']).open('rb') as stream:
                return pickle.load(stream)
        p1, f1 = metadata('reference')
        p2, f2 = metadata('repeated')
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal([p.K for p in p1], [p.K for p in p2])

    def test_legacy_missing_and_truncated_files(self):
        missing = {1, 17, 44, 82}
        corrupt = {3, 21, 67}
        saved = set(range(84))-missing
        dest = self.legacy('damaged', sorted(saved, reverse=True))
        for index in corrupt:
            (dest/f'_Kp-{index}.pickle').write_bytes(b'\x80\x05')
        calls = []
        with self.record_computation(calls):
            self.equal(self.run_calc('damaged', restart=True, restart_recover=True,
                                     restart_on_corrupt='recompute'))
        self.assert_only_missing(calls, saved-corrupt)
        self.assertEqual(len(list(dest.glob('*.corrupt-*'))), len(corrupt))

    def test_all_first_round_results_saved_before_completion_marker(self):
        with patch.object(Checkpoint, 'complete', side_effect=RuntimeError('completion marker')):
            with self.assertRaisesRegex(RuntimeError, 'completion marker'):
                self.run_calc('unmarked')
        with patch.object(runmod, 'get_data_k', side_effect=AssertionError('duplicate')):
            self.equal(self.run_calc('unmarked', restart=True))

    def test_initial_metadata_and_manifest_publication_interruptions(self):
        restart = importlib.import_module('wannierberri.restart')
        replace = restart.os.replace
        for boundary in ('metadata', 'manifest'):
            with self.subTest(boundary=boundary):
                def interrupted(source, destination):
                    path = Path(destination)
                    if ((boundary == 'metadata' and path.name.startswith('K_list-')) or
                            (boundary == 'manifest' and path.name == 'restart.json' and path.exists())):
                        raise RuntimeError('initial publication')
                    return replace(source, destination)
                name = 'publication-'+boundary
                with patch.object(restart.os, 'replace', side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, 'initial publication'):
                        self.run_calc(name)
                manifest = json.loads((self.root/name/'restart.json').read_text())
                self.assertEqual(manifest['phase'], 'initializing')
                calls = []
                with self.record_computation(calls):
                    self.equal(self.run_calc(name, restart=True))
                self.assert_only_missing(calls, set())


if __name__ == '__main__':
    unittest.main()
