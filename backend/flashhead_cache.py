"""Transactional RGB/state cache for bounded FlashHead rendering and recovery.

Only a final atomic commit record makes a chunk usable. Interrupted writes remain
uncommitted; source/settings changes and corrupted frame/state files are refused.
State is opaque here; the renderer must load its own tensors with weights_only.
"""
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):value.update(block)
    return value.hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,allow_nan=False))
    temporary.replace(path)


class ChunkCache:
    def __init__(self,directory,identity,shape=(28,512,512,3)):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.shape=tuple(shape);self.records=[]
        self.lock=(self.directory/'.cache-lock').open('a')
        try:
            fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            identity=dict(request=identity,geometry=list(self.shape),schema=1)
            canonical=json.dumps(identity,sort_keys=True,allow_nan=False).encode()
            self.fingerprint=hashlib.sha256(canonical).hexdigest()
            path=self.directory/'identity.json'
            if path.exists():
                if json.loads(path.read_text())!=identity:raise ValueError('Cached render inputs or settings changed; start a new render')
            else:
                if list(self.directory.glob('commit-*.json')):raise ValueError('Cache identity is missing')
                atomic_json(path,identity)
            commits=sorted(self.directory.glob('commit-*.json'))
            if [path.name for path in commits]!=[f'commit-{i:04d}.json' for i in range(len(commits))]:
                raise ValueError('Cache commit sequence has a gap')
            for index,path in enumerate(commits):
                record=json.loads(path.read_text());self.verify(index,record);self.records.append(record)
        except BaseException:
            self.close();raise

    def paths(self,index):
        return self.directory/f'chunk-{index:04d}.npy',self.directory/f'chunk-{index:04d}.state'

    def verify(self,index,record=None):
        record=self.records[index] if record is None else record
        frames,state=self.paths(index)
        if record.get('request')!=self.fingerprint or record.get('index')!=index:
            raise ValueError('Cached chunk identity differs')
        if type(record.get('append_frames')) is not int or not 0<record['append_frames']<=self.shape[0]:
            raise ValueError('Invalid cached output coverage')
        if sha(frames)!=record['frames_sha256'] or sha(state)!=record['state_sha256']:
            raise ValueError('Cached RGB frames or recovery state failed verification')
        if not 0<state.stat().st_size<=4*1024*1024:raise ValueError('Invalid recovery-state size')
        values=np.load(frames,mmap_mode='r',allow_pickle=False)
        if values.shape!=self.shape or values.dtype!=np.uint8:raise ValueError('Invalid cached RGB geometry')
        return record

    def commit(self,index,frames,state,append_frames):
        if index!=len(self.records):raise ValueError('Commit must extend the contiguous timeline')
        if frames.shape!=self.shape or frames.dtype!=np.uint8:raise ValueError('Invalid generated RGB geometry')
        if type(append_frames) is not int or not 0<append_frames<=self.shape[0]:raise ValueError('Invalid output coverage')
        if not isinstance(state,bytes) or not 0<len(state)<=4*1024*1024:raise ValueError('Invalid recovery state')
        frame_path,state_path=self.paths(index)
        temporary=frame_path.with_suffix('.tmp')
        with temporary.open('wb') as stream:np.save(stream,frames,allow_pickle=False)
        temporary.replace(frame_path)
        temporary=state_path.with_suffix('.tmp');temporary.write_bytes(state);temporary.replace(state_path)
        record=dict(request=self.fingerprint,index=index,append_frames=append_frames,
                    frames_sha256=sha(frame_path),state_sha256=sha(state_path))
        # Commit LAST: a crash before this point cannot make partial state usable.
        atomic_json(self.directory/f'commit-{index:04d}.json',record)
        self.records.append(record)

    def close(self):
        if self.lock is not None:self.lock.close();self.lock=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
