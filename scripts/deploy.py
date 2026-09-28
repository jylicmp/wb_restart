#!/usr/bin/env python3
"""Copy a committed checkout into the isolated environment (no editable install)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--environment', type=Path, required=True)
    p.add_argument('--allow-dirty', action='store_true', help='Development tests only; recorded in deployment manifest')
    args = p.parse_args()
    env = args.environment.resolve()
    if env.name != 'wberri_v1.8_restart':
        p.error('Deployment is restricted to wberri_v1.8_restart')
    root = Path(__file__).resolve().parents[1]
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip())
    if dirty and not args.allow_dirty:
        p.error('Commit changes first (or explicitly use --allow-dirty for development)')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    target = env/'lib/python3.12/site-packages/wannierberri'
    if not target.is_dir() or target.is_symlink():
        p.error('Expected physical WannierBerri package directory is missing')
    files = {}
    for source in sorted((root/'wannierberri').rglob('*.py')):
        rel = source.relative_to(root/'wannierberri')
        dest = target/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.parent.resolve() != dest.parent:
            p.error('Refusing a symlinked target directory')
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix='.deploy-')
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(source.read_bytes())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        files[str(rel)] = sha(dest)
        assert files[str(rel)] == sha(source)
    # Timestamp/size based .pyc caches from the cloned installation must not win.
    for cache in target.rglob('*.pyc'):
        cache.unlink()
    record = {'commit': commit, 'dirty': dirty, 'sha256': files}
    (target/'.wb_restart_deployment.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'commit': commit, 'dirty': dirty, 'files': len(files)}))


if __name__ == '__main__':
    main()
