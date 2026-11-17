#!/usr/bin/env python3
"""Pinned local setup; public downloads only, no media uploads or quality claim."""
import argparse,hashlib,json,platform,shutil,subprocess,sys,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true')
    p.add_argument('--skip-model-download',action='store_true');args=p.parse_args()
    lock=json.loads((ROOT/'research/flashhead-source-lock.json').read_text())
    source=ROOT/'vendor/SoulX-FlashHead';environment=ROOT/'.venv'
    manifests=[('flashhead-pro','flashhead-models'),('flashhead-audio','flashhead-audio')]
    if args.check_only:
        verified=all((source/name).is_file() and sha(source/name)==digest for name,digest in lock['vendor_files'].items())
        print(json.dumps(dict(source_verified=verified,environment_exists=(environment/'bin/python').is_file(),
            public_model_bytes=sum(item['bytes'] for name,_ in manifests for item in json.loads((ROOT/'backend/model_manifests'/f'{name}.json').read_text())['files']),
            visual_quality_approved=False,generated_video_verified=False),indent=2));return
    if sys.platform!='darwin' or platform.machine()!='arm64':raise RuntimeError('Use an Apple Silicon Mac')
    if int(platform.mac_ver()[0].split('.')[0])<14:raise RuntimeError('The pinned Torch runtime requires macOS 14 or later')
    for name,digest in lock['vendor_files'].items():
        target=source/name
        if target.exists():
            if sha(target)!=digest:raise ValueError('Pinned source differs; existing file retained: '+name)
            continue
        url=f'https://raw.githubusercontent.com/{lock["repository"]}/{lock["commit"]}/{name}'
        with urllib.request.urlopen(url,timeout=60) as response:data=response.read(4*1024*1024)
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Source checksum failed: '+name)
        target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix(target.suffix+'.partial')
        temporary.write_bytes(data);temporary.replace(target)
    if not environment.exists():
        python=shutil.which('python3.12')
        if not python:raise RuntimeError('Install Python 3.12 before setting up the local runtime')
        subprocess.run([python,'-m','venv',str(environment)],check=True)
    python=str(environment/'bin/python')
    subprocess.run([python,'-m','pip','install','-r',str(ROOT/'backend/requirements.lock.txt')],check=True)
    subprocess.run([python,'-c','import torch; assert torch.backends.mps.is_available(); x=torch.ones((8,8),device="mps"); y=x@x; torch.mps.synchronize(); assert float(y[0,0])==8; print("Metal test passed")'],check=True)
    if not args.skip_model_download:
        for name,folder in manifests:
            subprocess.run([python,str(ROOT/'research/download_model_manifest.py'),'--manifest',
                str(ROOT/'backend/model_manifests'/f'{name}.json'),'--output',str(ROOT/'.local'/folder)],check=True)
    print('Local setup finished. Run diagnostics and review actual generated video before using the renderer.')

if __name__=='__main__':main()
