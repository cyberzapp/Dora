"""Download a pinned HF manifest with bounded ranges and verified LFS hashes."""
import argparse,concurrent.futures,hashlib,json,os,time,subprocess,tempfile,re,fcntl
from pathlib import Path


def fetch(url, start=None, end=None, total=None, timeout=120):
    """Bound total wall time, including DNS/redirects, with native curl."""
    with tempfile.TemporaryDirectory(prefix='avatar-download-') as temporary:
        body=Path(temporary)/'body';headers=Path(temporary)/'headers'
        command=['curl','--silent','--show-error','--location','--fail',
                 '--connect-timeout','15','--max-time',str(timeout),
                 '--output',str(body),'--dump-header',str(headers)]
        if start is not None:command.extend(['--range',f'{start}-{end}'])
        result=subprocess.run(command+[url],capture_output=True,timeout=timeout+10)
        if result.returncode:
            # Do not log signed redirect URLs or authentication headers.
            raise RuntimeError(f'Weight transfer failed (curl exit {result.returncode})')
        if start is not None:
            ranges=re.findall(r'^content-range:\s*(.*?)\s*$',headers.read_text(),re.I|re.M)
            if not ranges or ranges[-1]!=f'bytes {start}-{end}/{total}':
                raise RuntimeError('Incorrect byte range')
        return body.read_bytes()


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        while data:=stream.read(8*1024*1024):value.update(data)
    return value.hexdigest()


def matches(path,size,expected_hash,blob_hash=None):
    if path.stat().st_size!=size or (expected_hash and digest(path)!=expected_hash):return False
    if blob_hash:
        value=hashlib.sha1(f'blob {size}\0'.encode())  # Hugging Face Git blob ID, not an LFS checksum.
        with path.open('rb') as stream:
            while data:=stream.read(1024*1024):value.update(data)
        if value.hexdigest()!=blob_hash:return False
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--publish-to',help='Atomically link verified completed files into a second model folder')
    parser.add_argument('--workers',type=int,choices=range(1,13),default=6,metavar='1..12',help='Concurrent range requests')
    parser.add_argument('--request-timeout',type=int,choices=range(30,601),default=120,metavar='30..600',help='Maximum seconds per transfer')
    args=parser.parse_args();manifest=json.loads(Path(args.manifest).read_text());folder=Path(args.output).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    # Keep this descriptor live for the entire download, including hash checks.
    lock_stream=(folder/'.download.lock').open('a')
    try:fcntl.flock(lock_stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another model download is already active for this folder')
    (folder/'download-manifest.json').write_text(json.dumps(manifest,indent=2))
    def publish(target,name,size,expected_hash,blob_hash):
        if not args.publish_to:return
        destination=Path(args.publish_to).resolve()/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(target,destination)
        except FileExistsError:
            if not matches(destination,size,expected_hash,blob_hash):
                raise RuntimeError('Existing published file failed validation: '+name)
        print('Published verified '+name,flush=True)
    for item in manifest['files']:
        name=item['path'];size=item['bytes'];expected_hash=item.get('sha256');blob_hash=item.get('git_blob_sha1')
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe manifest path')
        target=folder/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if matches(target,size,expected_hash,blob_hash):
                print('Verified existing '+name,flush=True);publish(target,name,size,expected_hash,blob_hash);continue
            raise RuntimeError('Existing file failed validation: '+name)
        url=f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{name}"
        part=target.with_name(target.name+'.partial');state=target.with_name(target.name+'.ranges.json')
        if size<1024*1024 or (blob_hash and size<16*1024*1024):
            part.write_bytes(fetch(url,timeout=args.request_timeout))
        else:
            block=8*1024*1024;count=(size+block-1)//block
            completed=set(json.loads(state.read_text())) if state.exists() else set()
            if completed and (not part.exists() or part.stat().st_size!=size):raise RuntimeError('Incomplete range state: '+name)
            fd=os.open(part,os.O_CREAT|os.O_RDWR,0o600);os.ftruncate(fd,size)
            def download(index):
                start=index*block;end=min(size,start+block)-1
                for attempt in range(6):
                    try:
                        content=fetch(url+f'?download=true&part={index}',start,end,size,timeout=args.request_timeout)
                        if len(content)!=end-start+1:raise RuntimeError('Truncated byte range')
                        os.pwrite(fd,content,start);return index
                    except Exception as error:
                        print(f'{name} range {index} attempt {attempt+1}: {error}',flush=True)
                        if attempt==5:raise
                        time.sleep(min(2**attempt,16))
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures=[pool.submit(download,index) for index in range(count) if index not in completed]
                    for future in concurrent.futures.as_completed(futures):
                        completed.add(future.result())
                        temporary=state.with_suffix('.tmp');temporary.write_text(json.dumps(sorted(completed)));temporary.replace(state)
                        if len(completed)%max(1,count//20)==0:print(f'{name}: {len(completed)/count:.0%}',flush=True)
            finally:os.close(fd)
        if not matches(part,size,expected_hash,blob_hash):
            raise RuntimeError('Integrity verification failed: '+name)
        part.replace(target);print('Verified '+name,flush=True)
        publish(target,name,size,expected_hash,blob_hash)
    # A waiting offline renderer must never observe a partially written marker.
    marker=folder/'download-complete.tmp'
    marker.write_text(json.dumps({'repo':manifest['repo'],'revision':manifest['revision'],'verified':True},indent=2))
    marker.replace(folder/'download-complete.json')


if __name__=='__main__':main()
