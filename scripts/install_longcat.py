#!/usr/bin/env python3
"""Reproducible local experimental LongCat installation; never uploads media."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vendor/longcat-avatar-mlx'
ENV = ROOT / '.venv-longcat'
MANIFEST = ROOT / 'backend/model_manifests/longcat-q4.json'


def source_verified(lock):
    return all((SOURCE / name).is_file() and
               hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
               for name, digest in lock['files'].items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true', help='Inspect setup without modifying files or accessing the network')
    parser.add_argument('--skip-model-download', action='store_true')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'research/longcat-source-lock.json').read_text())
    model = json.loads(MANIFEST.read_text())
    if args.check_only:
        print(json.dumps(dict(source_verified=source_verified(lock),
                              python_environment_exists=(ENV / 'bin/python').is_file(),
                              model_revision=model['revision'],
                              model_download_bytes=sum(x['bytes'] for x in model['files']),
                              generated_video_verified=False,
                              note='Installation state does not establish video quality'), indent=2))
        return
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        raise RuntimeError('This runtime requires an Apple Silicon Mac')
    if int(platform.mac_ver()[0].split('.')[0]) < 26:
        raise RuntimeError('The pinned LongCat Metal runtime requires macOS 26 or later. Update macOS before installing this model runtime.')
    python = shutil.which('python3.12')
    if not python:
        raise RuntimeError('Install Python 3.12, then run this installer again')
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', lock['repo'], str(SOURCE)], check=True)
        subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', lock['revision']], check=True)
    if not source_verified(lock):
        raise RuntimeError('The LongCat source differs from the pinned version. Preserve your edits and restore the pinned source before installing.')
    if not ENV.exists():
        subprocess.run([python, '-m', 'venv', str(ENV)], check=True)
    runtime = str(ENV / 'bin/python')
    subprocess.run([runtime, '-m', 'pip', 'install', '-r', str(ROOT / 'research/longcat-runtime-lock.txt')], check=True)
    subprocess.run([runtime, '-c', 'import mlx.core as mx; x=mx.ones((8,8)); assert float((x@x)[0,0])==8; print("Metal computation passed")'], check=True)
    if not args.skip_model_download:
        subprocess.run([runtime, str(ROOT / 'research/download_model_manifest.py'),
                        '--manifest', str(MANIFEST), '--output', str(ROOT / '.local/longcat-models/q4')], check=True)
    print('Local runtime setup finished. Real video generation and visual acceptance remain separate checks.')


if __name__ == '__main__':
    main()
