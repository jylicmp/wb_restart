"""Versioned, atomic checkpoints for the installed WannierBerri 1.8 API.

Pickles are executable Python objects: only load checkpoints from trusted runs.
A legacy result has no coordinate/configuration provenance; recovery is explicit.
"""
import copy
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import tempfile
import uuid
import warnings

import numpy as np

FORMAT = 1


class RestartError(RuntimeError):
    pass


class IncompatibleCheckpoint(RestartError):
    pass


class CorruptCheckpoint(RestartError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    """Address-independent descriptions; large numerical arrays are hashed."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('Object arrays cannot be fingerprinted')
        h = hashlib.sha256()
        # Bounded temporary storage, including non-contiguous input matrices.
        for chunk in np.nditer(value, flags=['external_loop', 'buffered', 'zerosize_ok'],
                               order='C', buffersize=1024 * 1024):
            h.update(chunk.tobytes())
        return {'array': h.hexdigest(), 'shape': list(value.shape), 'dtype': value.dtype.str}
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((canonical(v) for v in value), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(value, type):
        return {'class': value.__module__ + '.' + value.__qualname__}
    if hasattr(value, '__dict__') and not inspect.isroutine(value):
        return {'class': type(value).__module__ + '.' + type(value).__qualname__,
                'state': canonical(vars(value))}
    raise TypeError(f'Cannot fingerprint {type(value)}; checkpoint configuration must be explicit')


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def configuration(system, grid, calculators, parameters_K, use_irred_kpt, symmetrize, adpt_mesh, adpt_fac):
    if not hasattr(system, '_XX_R'):
        raise TypeError('Persistent restart currently requires a System_R (including System_tb)')
    names = ('real_lattice', 'periodic', 'num_wann', 'wannier_centers_cart', 'frozen_max',
             'force_internal_terms_only', 'is_phonon', 'spinor')
    state = {n: getattr(system, n, None) for n in names}
    state['matrices'] = system._XX_R
    state['rvectors'] = {n: getattr(system.rvec, n, None) for n in
                        ('iRvec', 'shifts_left_red', 'shifts_right_red', 'dim')}
    state['pointgroup'] = system.pointgroup
    # Freeze scientific implementation, excluding restart/dispatch-only edits.
    root = Path(__file__).parent
    code = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob('*.py'))
            if str(p.relative_to(root)) not in ('restart.py', 'run.py', 'grid/Kpoint.py')}
    return canonical({'system': state, 'grid': grid, 'calculators': calculators,
                      'parameters_K': parameters_K, 'use_irred_kpt': use_irred_kpt,
                      'symmetrize': symmetrize, 'adpt_mesh': adpt_mesh, 'adpt_fac': adpt_fac,
                      'scientific_source': code})


def atomic_write(path, writer):
    """Flush file, atomically rename on the same filesystem, then sync directory."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_pickle(path, value):
    atomic_write(path, lambda f: pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL))


def locked_run(func):
    """Lock lifetime includes initialization, result writes and refinement."""
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        b = bound.arguments
        from .grid import Path as GridPath
        enabled = (b['restart'] or b['allow_restart'] or b['dump_results']) and not isinstance(b['grid'], GridPath)
        if not enabled:
            return func(*args, **kwargs)
        import fcntl
        path = Path(b['file_Klist_path'] or '_tmp_wb')
        if b['restart'] and not path.is_dir():
            raise FileNotFoundError(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / '.restart.lock', 'a+b') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RestartError(f'Another restart writer holds {path}') from exc
            # Never unlink the lock file: replacing its inode defeats locking.
            return func(*args, **kwargs)
    return wrapper


def point_identity(kp):
    return digest({n: getattr(kp, n, None) for n in
                   ('K', 'dK', 'NKFFT', 'refinement_level', 'vertices', 'basis', 'split_level')})


def dump_point(kp):
    payload = pickle.dumps(kp.result, protocol=pickle.HIGHEST_PROTOCOL)
    envelope = {'wb_restart_format': FORMAT, 'index': kp.restart_index,
                'point': kp.restart_point, 'configuration': kp.restart_configuration,
                'sha256': hashlib.sha256(payload).hexdigest(), 'payload': payload}
    atomic_pickle(kp.result_storage_path, envelope)


def load_point(kp):
    try:
        with open(kp.result_storage_path, 'rb') as f:
            obj = pickle.load(f)
            if f.read(1):
                raise CorruptCheckpoint('Trailing bytes in result file')
        if isinstance(obj, dict) and 'wb_restart_format' in obj:
            if obj.get('wb_restart_format') != FORMAT:
                raise IncompatibleCheckpoint('Unsupported result format')
            if (obj.get('index') != kp.restart_index or obj.get('point') != kp.restart_point or
                    obj.get('configuration') != kp.restart_configuration):
                raise IncompatibleCheckpoint(f'Result identity/configuration mismatch: {kp.result_storage_path}')
            payload = obj['payload']
            if hashlib.sha256(payload).hexdigest() != obj['sha256']:
                raise CorruptCheckpoint('Result checksum mismatch')
            return pickle.loads(payload)
        if not getattr(kp, 'restart_legacy', False):
            raise IncompatibleCheckpoint('Unidentified legacy result in a new checkpoint')
        return obj
    except (EOFError, pickle.UnpicklingError, KeyError) as exc:
        raise CorruptCheckpoint(f'Incomplete result: {kp.result_storage_path}') from exc


def validate_result(result, calculators):
    from .result import ResultDict, EnergyResult
    from .smoother import VoidSmoother
    if not isinstance(result, ResultDict) or set(result.results) != set(calculators):
        raise IncompatibleCheckpoint('Calculator keys or result class do not match')
    schema = {}
    for key, calc in calculators.items():
        res = result.results[key]
        if isinstance(res, EnergyResult):
            data = res.data
            if not np.all(np.isfinite(data)):
                raise CorruptCheckpoint(f'Non-finite tensor: {key}')
            if hasattr(calc, 'Efermi'):
                if len(res.Energies) < 1 or not np.array_equal(res.Energies[0], calc.Efermi):
                    raise IncompatibleCheckpoint(f'Fermi energies differ: {key}')
                smoother = getattr(calc, 'smoother', None)
                smoother = VoidSmoother() if smoother is None else smoother
                if canonical(res.smoothers[0]) != canonical(smoother):
                    raise IncompatibleCheckpoint(f'Smoother differs: {key}')
            if type(calc).__name__.startswith('Qorb_GaoXiao') and res.rank != 2:
                raise IncompatibleCheckpoint(f'MQM tensor must have rank 2: {key}')
            if data.shape != tuple(len(e) for e in res.Energies) + (3,) * res.rank:
                raise IncompatibleCheckpoint(f'Tensor shape differs: {key}')
            schema[key] = canonical({'shape': data.shape, 'energies': res.Energies,
                                     'TR': res.transformTR, 'Inv': res.transformInv})
        else:
            # Existing generic result classes remain supported; strict scientific
            # validation above is specifically for the EnergyResult MQM use case.
            schema[key] = canonical(type(res))
    return schema


def choose_iteration(indices, requested):
    indices = sorted(set(int(i) for i in indices))
    if not indices:
        raise CorruptCheckpoint('No iteration weights/checkpoints found')
    if requested < 0:
        target = max(0, indices[-1] + requested + 1)
        earlier = [i for i in indices if i <= target]
        if not earlier:
            raise RestartError(f'No checkpoint at or before iteration {target}')
        return earlier[-1]
    if requested not in indices:
        raise RestartError(f'Iteration {requested} is unavailable; available: {indices}')
    return requested


class Checkpoint:
    def __init__(self, path, config, calculators):
        self.path = Path(path)
        self.config = config
        self.fingerprint = digest(config)
        self.calculators = calculators
        self.manifest_path = self.path / 'restart.json'
        self.manifest = {'format': FORMAT, 'configuration': config,
                         'fingerprint': self.fingerprint, 'iterations': {}}
        self.schema = None

    def fresh(self):
        entries = [p for p in self.path.iterdir() if p.name != '.restart.lock']
        if entries:
            raise RestartError(f'Refusing to erase nonempty checkpoint directory {self.path}; use restart=True or a new directory')

    def bind(self, points, legacy=False):
        for i, kp in enumerate(points):
            kp.restart_index = i
            kp.restart_point = point_identity(kp)
            kp.restart_configuration = self.fingerprint
            if legacy:
                kp.restart_legacy = True
                filename = f'_Kp-{i}.pickle'
            elif getattr(kp, 'result_storage_path', None):
                filename = Path(kp.result_storage_path).name
            else:
                filename = f'_Kp-{i}-{kp.restart_point[:20]}.pickle'
                kp.restart_legacy = False
            kp.set_storage_path(str(self.path / filename))

    def load(self, grid, use_irred_kpt, requested=-1, recover=False):
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
            if self.manifest.get('format') != FORMAT:
                raise IncompatibleCheckpoint('Unsupported checkpoint format')
            if self.manifest['fingerprint'] != self.fingerprint or self.manifest['configuration'] != self.config:
                raise IncompatibleCheckpoint('Model, grid, calculator, refinement or scientific source changed')
            it = choose_iteration(self.manifest['iterations'], requested)
            entry = self.manifest['iterations'][str(it)]
            filename = entry['metadata']
            if Path(filename).name != filename:
                raise CorruptCheckpoint('Invalid metadata filename')
            path = self.path / filename
            if sha256_file(path) != entry['sha256']:
                raise CorruptCheckpoint('K-point metadata checksum mismatch')
            with path.open('rb') as f:
                points, factors = pickle.load(f)
            self.schema = self.manifest.get('result_schema')
            legacy = False
        else:
            files = {int(p.stem.split('-')[-1]): p for p in self.path.glob('factors_iter-*.npy')}
            it = choose_iteration(files, requested)
            factors = np.load(files[it], allow_pickle=False)
            kfile = self.path / 'K_list.pickle'
            if kfile.exists():
                points = []
                with kfile.open('rb') as f:
                    while f.peek(1) if hasattr(f, 'peek') else False:
                        points.extend(pickle.load(f))
                if len(points) < len(factors):
                    raise CorruptCheckpoint('Legacy K-list shorter than weights')
                points = points[:len(factors)]
            else:
                if not recover:
                    raise RestartError('K_list.pickle missing; explicit restart_recover=True is required')
                from .grid import Grid
                if not isinstance(grid, Grid) or set(files) != {0} or it != 0:
                    raise RestartError('Missing K-list recovery is supported only for a regular-grid iteration 0')
                points = grid.get_K_list(use_symmetry=use_irred_kpt)
                expected = np.array([p.factor for p in points])
                if factors.shape != expected.shape or not np.array_equal(factors, expected):
                    raise IncompatibleCheckpoint('Regenerated point count or per-index weights differ')
            warnings.warn('Legacy results have no coordinate/configuration provenance; validate original inputs and sample recomputations')
            self.manifest['legacy_provenance'] = 'Original input reconstruction; raw results lack coordinate/configuration hashes'
            legacy = True
        if factors.shape != (len(points),) or not np.all(np.isfinite(factors)) or np.any(factors < 0):
            raise CorruptCheckpoint('Invalid weight vector')
        if not np.isclose(factors.sum(), 1., rtol=0, atol=1e-10):
            raise CorruptCheckpoint('Weights do not sum to one')
        for kp, factor in zip(points, factors):
            kp.set_factor(float(factor))
        self.bind(points, legacy=legacy)
        return it, points, factors

    def restore(self, points, on_corrupt='error', readonly=False):
        if on_corrupt not in ('error', 'recompute'):
            raise ValueError('restart_on_corrupt must be error or recompute')
        report = {'valid': 0, 'missing': [], 'corrupt': [], 'weight_sum': float(sum(k.factor for k in points))}
        total = None
        for i, kp in enumerate(points):
            kp.result = None
            kp.was_evaluated_flag = False
            kp.res_dumped_flag = False
            kp.res_cleared_flag = False
            try:
                result = load_point(kp)
                schema = validate_result(result, self.calculators)
                if self.schema is None:
                    self.schema = schema
                elif self.schema != schema:
                    raise IncompatibleCheckpoint(f'Result tensor schema differs at K point {i}')
            except FileNotFoundError:
                report['missing'].append(i)
                continue
            except CorruptCheckpoint:
                report['corrupt'].append(i)
                if on_corrupt == 'error' and not readonly:
                    raise
                if not readonly:
                    os.replace(kp.result_storage_path, kp.result_storage_path + '.corrupt-' + uuid.uuid4().hex)
                continue
            kp.set_result(result)
            total = total + kp.get_result_factor() if total is not None else kp.get_result_factor()
            kp.result = None
            kp.res_dumped_flag = True
            report['valid'] += 1
        print(f"Restart: reused={report['valid']}, missing={len(report['missing'])}, corrupt={len(report['corrupt'])}", flush=True)
        return total, report

    def begin(self, iteration, points):
        self.bind(points)
        metadata = []
        for kp in points:
            item = copy.copy(kp)
            item.result = None
            item.was_evaluated_flag = False
            item.res_dumped_flag = False
            item.res_cleared_flag = False
            # Derived arrays can be large and are reconstructed when needed.
            for name in ('star', 'distGamma', '_max'):
                item.__dict__.pop(name, None)
            item.result_storage_path = Path(item.result_storage_path).name
            metadata.append(item)
        factors = np.array([float(kp.factor) for kp in points])
        name = f'K_list-{iteration:08d}-{uuid.uuid4().hex}.pickle'
        atomic_pickle(self.path / name, (metadata, factors))
        # A historical restart starts a new branch; later iterations are no
        # longer selected, but their immutable files are left intact.
        self.manifest['iterations'] = {i: e for i, e in self.manifest['iterations'].items() if int(i) < iteration}
        self.manifest['iterations'][str(iteration)] = {'metadata': name, 'sha256': sha256_file(self.path / name),
                                                       'phase': 'running', 'points': len(points)}
        self._commit()

    def complete(self, iteration):
        self.manifest['iterations'][str(iteration)]['phase'] = 'complete'
        self._commit()

    def _commit(self):
        self.manifest['result_schema'] = self.schema
        data = json.dumps(self.manifest, sort_keys=True, indent=2).encode()
        atomic_write(self.manifest_path, lambda f: f.write(data))
