"""Download the pinned official Dub-It adapter using an in-memory read token.

The token is entered through an echo-disabled terminal prompt, never persisted,
and sent only to huggingface.co. This does not accept model access agreements.
"""
import fcntl
import getpass
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT/'research/dubit-source-lock.json').read_text())
    item = lock['files'][0]
    directory = ROOT/'.local/ltx-dubit-weights'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'download.lock').open('a') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = directory/item['path']
        def digest(path):
            result = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024**2), b''): result.update(block)
            return result.hexdigest()
        if target.exists():
            if target.stat().st_size == item['bytes'] and digest(target) == item['sha256']:
                print('Pinned official adapter already present and verified.', flush=True)
                return
            raise ValueError('Existing adapter failed verification; retain it for inspection')
        if shutil.disk_usage(directory).free < item['bytes']+10*1024**3:
            raise ValueError('Leave room for the adapter and 10 GiB for rendering/swap')
        token = getpass.getpass('Read-only Hugging Face token (not saved): ').strip()
        if not token: raise ValueError('No token supplied')
        url = f"https://huggingface.co/{lock['repo']}/resolve/{lock['revision']}/{item['path']}"
        partial = target.with_suffix(target.suffix+'.partial')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item['bytes']: raise ValueError('Partial exceeds the expected pinned file size')
        request = urllib.request.Request(url, headers={'User-Agent':'AvatarStudio-local-installer'})
        # urllib does not copy unredirected headers to a redirected CDN request.
        request.add_unredirected_header('Authorization', 'Bearer '+token)
        if offset: request.add_header('Range', f'bytes={offset}-')
        started = time.monotonic()
        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as error:
            report = dict(status=error.code, authenticated_request=True, token_saved=False,
                          access_terms_accepted=False, repo=lock['repo'], revision=lock['revision'],
                          error_code=error.headers.get('X-Error-Code'))
            message = error.headers.get('X-Error-Message', '')
            report['access_reason'] = message.replace(token,'[redacted]')[:1000]
            identity_request = urllib.request.Request('https://huggingface.co/api/whoami-v2')
            identity_request.add_unredirected_header('Authorization','Bearer '+token)
            try:
                with urllib.request.urlopen(identity_request,timeout=30) as identity_response:
                    identity = json.load(identity_response)
                report['token_valid'] = True
                report['token_role'] = identity.get('auth',{}).get('accessToken',{}).get('role')
            except urllib.error.HTTPError as identity_error:
                report['token_valid'] = False
                report['token_check_status'] = identity_error.code
            finally:
                identity_request.remove_header('Authorization')
            (directory/'access.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
            return
        finally:
            token = None
            request.remove_header('Authorization')
        with response:
            if response.status == 206:
                if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('Server resumed the wrong range')
            elif response.status == 200:
                offset = 0
            else: raise ValueError('Unexpected model response status')
            downloaded = offset
            last_report = downloaded
            with partial.open('ab' if offset else 'wb') as stream:
                while block := response.read(8*1024**2):
                    downloaded += len(block)
                    if downloaded > item['bytes']: raise ValueError('Response exceeds the pinned model size')
                    stream.write(block)
                    if downloaded-last_report >= 64*1024**2:
                        state = dict(bytes=downloaded,total=item['bytes'],elapsed_seconds=time.monotonic()-started,
                                     verified=False,token_saved=False)
                        temporary = directory/'progress.tmp'
                        temporary.write_text(json.dumps(state,indent=2));temporary.replace(directory/'progress.json')
                        print(json.dumps(state),flush=True);last_report=downloaded
        if downloaded != item['bytes'] or digest(partial) != item['sha256']:
            raise ValueError('Official model download failed pinned size or SHA-256 verification')
        partial.replace(target)
        complete = dict(repo=lock['repo'],revision=lock['revision'],file=str(target),
                        bytes=item['bytes'],sha256=item['sha256'],verified=True,
                        token_saved=False,access_terms_accepted=False,elapsed_seconds=time.monotonic()-started)
        (directory/'download-complete.json').write_text(json.dumps(complete,indent=2))
        print(json.dumps(complete),flush=True)


if __name__ == '__main__': main()
