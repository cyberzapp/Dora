#!/usr/bin/env python3
"""Build an isolated relocatable local-AI runtime from verified local installs.

Packaging infrastructure only. Does not copy user assets, models, or project
metadata and does not establish model-generation or perceptual quality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

MAGIC = {bytes.fromhex(x) for x in ('cffaedfe', 'cefaedfe', 'feedfacf', 'feedface', 'cafebabe', 'bebafeca', 'cafebabf')}


def run(args):
    return subprocess.run([str(x) for x in args], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def native(path):
    if path.is_symlink() or not path.is_file(): return False
    with path.open('rb') as source: return source.read(4) in MAGIC


def linked(path):
    lines = run(['otool', '-L', path]).splitlines()
    return [line.strip().split(' (compatibility')[0] for line in lines if ' (compatibility' in line]


def identifier(path):
    lines = run(['otool', '-D', path]).splitlines()
    return lines[1].strip() if len(lines) > 1 and not lines[1].endswith(':') else None


def sha(path):
    with path.open('rb') as source: return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-engine', action='store_true', help='Include the selected engine code, never personal files or model weights')
    parser.add_argument('--profile', choices=['longcat','flashhead'],default='longcat')
    args = parser.parse_args()
    if sys.platform != 'darwin' or platform.machine() != 'arm64': raise RuntimeError('Run on the target Apple Silicon Mac')
    source_env = args.environment.resolve()
    base = Path(run([source_env/'bin/python', '-c', 'import sys;print(sys.base_prefix)']).strip()).resolve()
    if base.parent.name != 'Versions' or base.parent.parent.name != 'Python.framework':
        raise ValueError('Expected a framework Python installation')
    if base.name != '3.12': raise ValueError('Use the pinned Python 3.12 environment')
    source_framework = base.parent.parent
    root = args.output.resolve()
    if root.exists(): raise ValueError('Use a new output directory; existing artifacts are preserved')
    root.mkdir(parents=True)
    framework = root/'Python.framework'; environment = root/('.venv' if args.profile=='flashhead' else '.venv-longcat')
    run(['cp', '-cR', source_framework, framework])
    (environment/'lib/python3.12').mkdir(parents=True)
    run(['cp', '-cR', source_env/'lib/python3.12/site-packages', environment/'lib/python3.12/site-packages'])
    bundled_base = framework/'Versions'/base.name
    global_site = bundled_base/'lib/python3.12/site-packages'
    if global_site.is_symlink(): global_site.unlink(); global_site.mkdir()
    for path in framework.rglob('*'):
        if path.is_symlink() and path.is_absolute() and path.suffix == '.a': path.unlink()
    # No stale build paths or external interpreter location in the environment.
    (environment/'bin').mkdir()
    (environment/'bin/python3.12').symlink_to(os.path.relpath(bundled_base/'bin/python3.12', environment/'bin'))
    (environment/'pyvenv.cfg').write_text('include-system-site-packages = false\nversion = 3.12\n')
    launcher = '''#!/bin/sh
runtime_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P) || exit 1
export PYTHONHOME="$runtime_root/Python.framework/Versions/3.12"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
exec "$runtime_root/.venv-longcat/bin/python3.12" "$@"
'''
    launcher=launcher.replace('.venv-longcat',environment.name)
    (environment/'bin/python').write_text(launcher); (environment/'bin/python').chmod(0o755)
    libraries = root/'native-libs'; libraries.mkdir()
    notices = root/'licenses'; notices.mkdir()
    engine_files = {}
    if args.include_engine and args.profile=='longcat':
        project = Path(__file__).resolve().parents[1]
        lock = json.loads((project/'research/longcat-source-lock.json').read_text())
        files = ['backend/worker.py', 'backend/inference_lease.py', 'backend/longcat_render.py',
                 'backend/longcat_timeline.py', 'backend/head_composite.py', 'backend/render_storage.py',
                 'backend/longcat_doctor.py', 'backend/model_manifests/longcat-q4.json',
                 'research/longcat_audio_probe.py', 'research/longcat_attention.py',
                 'research/longcat_benchmark.py', 'research/longcat_sampler.py',
                 'research/longcat_streaming.py', 'research/longcat-source-lock.json', 'scripts/verify_export.py']
        for name, expected in lock['files'].items():
            relative = 'vendor/longcat-avatar-mlx/'+name
            if sha(project/relative) != expected: raise ValueError('Vendor source failed pinned verification: '+name)
            files.append(relative)
        for name in files:
            target = root/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(project/name,target);engine_files[name]=sha(target)
        for name, label in [('vendor/longcat-avatar-mlx/LICENSE','LongCat-MLX-LICENSE.txt'),('vendor/LongCat-Video/LICENSE','LongCat-Upstream-LICENSE.txt')]:
            shutil.copy2(project/name,notices/label)
    if args.include_engine and args.profile=='flashhead':
        project=Path(__file__).resolve().parents[1]
        lock=json.loads((project/'research/flashhead-source-lock.json').read_text())
        files=['backend/worker.py','backend/inference_lease.py','backend/flashhead_render.py',
               'backend/flashhead_engine.py','backend/flashhead_audio.py','backend/flashhead_cache.py',
               'backend/flashhead_timeline.py','backend/flashhead_layout.py','backend/head_composite.py',
               'backend/head_stabilization.py','backend/flashhead_motion.py','backend/flashhead_doctor.py',
               'backend/model_manifests/flashhead-pro.json','backend/model_manifests/flashhead-audio.json',
               'research/flashhead_mps.py','research/flashhead-source-lock.json','scripts/verify_export.py']
        for name,expected in lock['vendor_files'].items():
            relative='vendor/SoulX-FlashHead/'+name
            if sha(project/relative)!=expected:raise ValueError('Pinned FlashHead source changed: '+name)
            files.append(relative)
        for name in files:
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(project/name,target);engine_files[name]=sha(target)
        shutil.copy2(project/'vendor/SoulX-FlashHead/LICENSE',notices/'FlashHead-Apache-2.0.txt')
    (notices/'PYTHON-MODIFICATIONS.txt').write_text(
        'Python was copied from the local framework installation. Dynamic library paths were relocated '
        'and changed native binaries were ad-hoc signed. Static build-library links were omitted. '
        'A launch wrapper selects bundled standard libraries and disables user-site packages. '
        'The Python language and standard-library source were not modified. Original license notices '
        'are retained in this directory and dependency dist-info directories.\n')
    ffmpeg = next((environment/'lib/python3.12/site-packages/imageio_ffmpeg/binaries').glob('ffmpeg-macos-*'))
    license_result = subprocess.run([str(ffmpeg), '-L'], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (notices/'FFMPEG-BINARY-LICENSE.txt').write_text(license_result.stdout)
    (notices/'DISTRIBUTION-REVIEW.txt').write_text(
        'The FFmpeg binary reports GPL-enabled compilation. Retaining these notices does not by itself '
        'complete source-availability and redistribution obligations. Review all bundled model, wheel '
        'and codec licenses before distributing a release. This runtime is a local packaging test.\n')
    mappings = [(source_framework.resolve(), framework), (source_env, environment)]
    copied = {}; license_packages = set()

    def copy_licenses(source):
        parts = source.parts
        if 'Cellar' not in parts: return
        at = parts.index('Cellar')
        package = Path(*parts[:at+3])
        if package in license_packages: return
        license_packages.add(package)
        files = [p for p in package.iterdir() if p.is_file() and any(s in p.name.lower() for s in ('license', 'copying', 'copyright'))]
        for directory in [package/'share/doc', package/'share/licenses']:
            if directory.is_dir(): files.extend(p for p in directory.rglob('*') if p.is_file() and any(s in p.name.lower() for s in ('license', 'copying', 'copyright')))
        target = notices/(parts[at+1]+'-'+parts[at+2]); target.mkdir(exist_ok=True)
        for index, path in enumerate(files): shutil.copy2(path, target/f'{index}-{path.name}')
        if not files and parts[at+1] == 'sqlite':
            shutil.copy2(Path(__file__).resolve().parents[1]/'resources/RuntimeNotices/SQLite.txt', target/'SQLite.txt')
        elif not files: (target/'REVIEW-REQUIRED.txt').write_text('No license notice located automatically. Review dependency licensing before distribution.\n')

    def destination(dependency):
        source = Path(dependency).resolve(strict=True)
        for original, target in mappings:
            if source.is_relative_to(original): return target/source.relative_to(original)
        if not dependency.startswith('/opt/homebrew/'):
            raise ValueError('Unbundled non-system dependency: '+dependency)
        if source not in copied:
            target = libraries/(hashlib.sha256(str(source).encode()).hexdigest()[:12]+'-'+source.name)
            shutil.copy2(source, target); copied[source] = target
            copy_licenses(source)
        return copied[source]

    queue = [p for p in root.rglob('*') if native(p)]; seen = set(); dependencies = []; thinned=[]
    # Universal MediaPipe/protobuf wheels have different LC_ID_DYLIB names in
    # each slice. This artifact targets Apple Silicon only: select arm64 before
    # resolving identifiers, rather than mistaking the Intel ID for a dependency.
    for binary in queue:
        with binary.open('rb') as stream:magic=stream.read(4)
        if magic in {bytes.fromhex(x) for x in ('cafebabe','bebafeca','cafebabf')}:
            architectures=run(['lipo','-archs',binary]).split()
            if 'arm64' not in architectures:raise ValueError('No Apple Silicon slice: '+str(binary))
            temporary=binary.with_name(binary.name+'.arm64-tmp')
            run(['lipo',binary,'-thin','arm64','-output',temporary]);temporary.replace(binary)
            run(['codesign','--force','--sign','-',binary]);thinned.append(str(binary.relative_to(root)))
    copy_licenses(base/'Python')
    while queue:
        binary = queue.pop(0)
        if binary in seen: continue
        seen.add(binary); changed = False
        own_id = identifier(binary)
        if own_id and own_id.startswith('/'):
            run(['install_name_tool', '-id', '@rpath/'+binary.name, binary]); changed = True
        for dependency in linked(binary):
            if dependency == own_id or dependency.startswith(('@', '/usr/lib/', '/System/Library/')): continue
            target = destination(dependency)
            if not target.is_file(): raise FileNotFoundError(target)
            relative = '@loader_path/'+os.path.relpath(target, binary.parent)
            run(['install_name_tool', '-change', dependency, relative, binary]); changed = True
            dependencies.append(dict(binary=str(binary.relative_to(root)), original=dependency, bundled=str(target.relative_to(root))))
            if target not in seen: queue.append(target)
        if changed: run(['codesign', '--force', '--sign', '-', binary])
        if len(seen) % 50 == 0: print(f'Audited {len(seen)} native binaries', flush=True)
    external_links = []
    for path in root.rglob('*'):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            external_links.append(str(path.relative_to(root)))
    if external_links: raise ValueError('External symlinks remain: '+', '.join(external_links))
    unresolved = []
    for binary in seen:
        for dependency in linked(binary):
            if dependency.startswith('/') and not dependency.startswith(('/usr/lib/', '/System/Library/')):
                unresolved.append((str(binary.relative_to(root)), dependency))
    if unresolved: raise ValueError('External linked libraries remain: '+str(unresolved))
    report = dict(runtime_only=True, generated_video_verified=False, quality_approved=False,
                  engine_included=args.include_engine,engine_profile=args.profile,environment_name=environment.name,
                  models_included=False, engine_sha256=engine_files,
                  python=base.name, environment=str(source_env), native_binaries=len(seen),
                  relocated_dependencies=dependencies, external_absolute_dependencies=unresolved,
                  target_architecture='arm64',thinned_universal_binaries=thinned,
                  license_review_required=True,
                  missing_dependency_notices=[str(p.relative_to(root)) for p in notices.rglob('REVIEW-REQUIRED.txt')],
                  native_sha256={str(p.relative_to(root)):sha(p) for p in sorted(seen)})
    (root/'packaging-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key:value for key,value in report.items() if key not in ('native_sha256','relocated_dependencies','engine_sha256')},indent=2))


if __name__ == '__main__': main()
