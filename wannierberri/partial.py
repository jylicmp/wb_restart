"""Observation-only iteration-zero snapshots. Never a restart authority."""
import fcntl
import json
import os
from pathlib import Path
import time
import uuid
import warnings

import numpy as np
from .restart import atomic_write, IncompatibleCheckpoint, RestartError, sha256_file
from .result import EnergyResult


def validate_output(path, checkpoint=None):
    path = Path(path).resolve()
    protected = []
    if checkpoint is not None:
        protected.append(Path(checkpoint).resolve())
    if os.environ.get('WB_SOURCE'):
        protected.append(Path(os.environ['WB_SOURCE']).resolve().parent)
    for source in protected:
        if path == source or source in path.parents or path in source.parents:
            raise RestartError('Partial output must be separate from the checkpoint/protected task tree')
    return path


class PartialWriter:
    def __init__(self, path, fingerprint, points, interval=900, report=None, dynamic=False):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.points = points
        self.interval = float(interval)
        if not np.isfinite(self.interval) or self.interval < 0:
            raise ValueError('Partial interval must be finite and nonnegative')
        self.mask = np.array([p.was_evaluated_flag for p in points], dtype=bool)
        self.count = int(self.mask.sum())
        self.weight = float(sum(p.factor for p, valid in zip(points, self.mask) if valid))
        self.corrupt = set((report or {}).get('corrupt', []))
        self.unstable = set((report or {}).get('unstable', []))
        self.dynamic = dynamic
        self.last_count = -1
        self.last_attempt = None
        self.lock = None

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = (self.path/'.partial.lock').open('a+b')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity = self.path/'identity.json'
            if identity.exists():
                if json.loads(identity.read_text())['configuration'] != self.fingerprint:
                    raise IncompatibleCheckpoint('Partial output belongs to a different configuration')
            else:
                entries = [p for p in self.path.iterdir() if p.name != '.partial.lock']
                if entries:
                    raise RestartError('Refusing unrecognized nonempty partial output directory')
                atomic_write(identity, lambda f: f.write(json.dumps({'configuration': self.fingerprint}).encode()))
        except BaseException:
            self.lock.close()
            raise
        return self

    def __exit__(self, *exc):
        self.lock.close()

    def observe(self, point, subtotal, base=None):
        index = point.restart_index if point is not None else None
        if index is not None and not self.mask[index]:
            self.mask[index] = True
            self.count += 1
            self.weight += point.factor
            self.corrupt.discard(index)
            self.unstable.discard(index)
        if self.count == self.last_count:
            return
        now = time.monotonic()
        # First calculated point is useful immediately, including after an empty report.
        if self.last_attempt is not None and not (self.count == 1 and self.last_count == 0) and now-self.last_attempt < self.interval:
            return
        self.publish(subtotal if base is None else base+subtotal, strict=False)

    def publish(self, result, strict=True):
        if self.count == self.last_count:
            return
        self.last_attempt = time.monotonic()
        try:
            self._publish(result)
            self.last_count = self.count
        except Exception as exc:
            if strict:
                raise
            warnings.warn(f'Partial snapshot failed; calculation continues: {exc}', RuntimeWarning)

    def _publish(self, result):
        if result is not None and not all(isinstance(r, EnergyResult) for r in result.results.values()):
            raise TypeError('Partial snapshots support EnergyResult calculators only')
        name = 'snapshot-'+time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'-'+uuid.uuid4().hex
        directory = self.path/name
        directory.mkdir()
        arrays = {'completed_indices': np.flatnonzero(self.mask)}
        tensors = {}
        for number, (key, value) in enumerate((result.results if result is not None else {}).items()):
            label = f'c{number:04d}'
            arrays[label+'_data'] = value.data
            arrays[label+'_dataSmooth'] = value.dataSmooth
            for i, energy in enumerate(value.Energies):
                arrays[f'{label}_energy_{i}'] = energy
            tensors[key] = {'prefix': label, 'rank': int(value.rank), 'energy_titles': list(value.E_titles)}
            # Independent text writer: full precision, explicit raw/smoothed components.
            energy_columns = np.stack(np.meshgrid(*value.Energies, indexing='ij'), axis=-1)
            n = int(np.prod([len(e) for e in value.Energies]))
            data = value.data.reshape(n, -1)
            smooth = value.dataSmooth.reshape(n, -1)
            names = list(value.E_titles)
            cols = [energy_columns.reshape(n, -1)]
            for kind, values in (('raw', data), ('smooth', smooth)):
                for component in range(values.shape[1]):
                    v = values[:, component:component+1]
                    if np.iscomplexobj(v):
                        cols.extend([v.real, v.imag]); names.extend([f'{kind}_{component}_real', f'{kind}_{component}_imag'])
                    else:
                        cols.append(v); names.append(f'{kind}_{component}')
            header = 'Iteration 0 weighted contribution; no coverage renormalization\n'+' '.join(names)
            atomic_write(directory/(label+'.dat'), lambda f, c=cols, h=header: np.savetxt(f, np.concatenate(c, axis=1), fmt='%.17e', header=h))
        if result is not None:
            atomic_write(directory/'tensors.npz', lambda f: np.savez_compressed(f, **arrays))
        meta = {'format': 1, 'iteration': 0, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'configuration': self.fingerprint, 'complete': self.count == len(self.points),
                'completed_points': self.count, 'total_points': len(self.points),
                'completed_weight': self.weight, 'total_weight': float(sum(p.factor for p in self.points)),
                'missing_points': len(self.points)-self.count-len(self.corrupt)-len(self.unstable),
                'corrupt_points': len(self.corrupt), 'unstable_points': len(self.unstable),
                'dynamic_snapshot': self.dynamic, 'renormalized': False, 'calculators': tensors}
        meta['files'] = {p.name: sha256_file(p) for p in directory.iterdir()}
        atomic_write(directory/'metadata.json', lambda f: f.write(json.dumps(meta, indent=2).encode()))
        # Sync the generation directory entry before making it discoverable.
        fd = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        atomic_write(self.path/'latest.json', lambda f: f.write(json.dumps({'format': 1, 'snapshot': name}).encode()))
        print(f'Partial snapshot: {self.count}/{len(self.points)} points, weight={self.weight:.12g}, {directory}', flush=True)
