#!/usr/bin/env python3
"""Copy a live tree to a new independent directory, validating each stable file."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def signature(p):
    s=p.stat()
    return s.st_ino,s.st_size,s.st_mtime_ns


def checksum(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    a=parser.parse_args();source=a.source.resolve();target=a.target.resolve()
    if target==source or source in target.parents or target in source.parents:
        parser.error('Source and target must be separate trees')
    if a.workers<1:parser.error('workers must be positive')
    target.mkdir(exist_ok=False)
    control=target/'_restart_test';control.mkdir()
    started=time.time()
    state={'source':str(source),'target':str(target),'dynamic_snapshot':True,'started':started,'status':'copying'}
    (control/'copy-state.json').write_text(json.dumps(state,indent=2))
    files=[]
    for directory,subdirs,names in os.walk(source,followlinks=False):
        relative=Path(directory).relative_to(source)
        for name in subdirs:
            if (Path(directory)/name).is_symlink():raise RuntimeError('Directory symlinks require explicit handling')
            (target/relative/name).mkdir(exist_ok=True)
        files.extend(relative/name for name in names)
    print(f'Inventory: {len(files)} files; copying with {a.workers} workers',flush=True)
    def copy_one(relative):
        src=source/relative;dst=target/relative;temporary=dst.with_name('.'+dst.name+'.copy-tmp')
        for attempt in range(3):
            before=signature(src)
            digest=hashlib.sha256()
            with src.open('rb') as reader,temporary.open('wb') as writer:
                for block in iter(lambda:reader.read(8*1024**2),b''):
                    writer.write(block);digest.update(block)
            after=signature(src)
            if before!=after:
                temporary.unlink();time.sleep(.2);continue
            if checksum(temporary)!=digest.hexdigest():raise IOError(f'Copy checksum failed: {relative}')
            shutil.copystat(src,temporary,follow_symlinks=True)
            os.replace(temporary,dst)
            if src.stat().st_ino==dst.stat().st_ino or dst.is_symlink():raise IOError('Copy must be independent')
            return {'path':str(relative),'size':before[1],'sha256':digest.hexdigest(),
                    'source_mtime_ns':before[2],'stable_during_copy':True}
        raise IOError(f'Source did not stabilize in three attempts: {relative}')
    count=0;total=0;errors=[]
    with (control/'copy-files.jsonl').open('w') as manifest,ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending={};remaining=iter(files);exhausted=False
        while pending or not exhausted:
            while not exhausted and len(pending)<4*a.workers:
                try:relative=next(remaining)
                except StopIteration:exhausted=True;break
                pending[pool.submit(copy_one,relative)]=str(relative)
            if not pending:break
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                path=pending.pop(future)
                try:
                    record=future.result();total+=record['size'];count+=1
                    manifest.write(json.dumps(record)+'\n')
                except Exception as exc:
                    errors.append({'path':path,'error':str(exc)})
            if count and count%1000<len(ready):
                manifest.flush()
                print(f'Copied and verified: {count}/{len(files)} files, {total/1024**3:.2f} GiB, {time.time()-started:.0f}s',flush=True)
        manifest.flush();os.fsync(manifest.fileno())
    state.update(status='failed' if errors else 'complete',files=count,bytes=total,
                 inventory_files=len(files),ended=time.time(),errors=errors,
                 note='Per-file stable copies; not a globally frozen snapshot. Files created after inventory are not included.')
    (control/'copy-state.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(state,indent=2),flush=True)
    if errors:raise RuntimeError('Copy incomplete; do not start recovery')


if __name__=='__main__':main()
