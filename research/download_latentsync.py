"""Resumable, range-validated checkpoint download with final SHA-256 verification."""
from pathlib import Path
import concurrent.futures,hashlib,json,os,time
import requests
import argparse
parser=argparse.ArgumentParser();parser.add_argument("--version",choices=["1.5","1.6"],default="1.5");args=parser.parse_args()
from huggingface_hub import get_hf_file_metadata,hf_hub_url
root=Path(__file__).resolve().parents[1]
u=hf_hub_url('ByteDance/LatentSync-'+args.version,'latentsync_unet.pt')
m=get_hf_file_metadata(u);folder=root/('vendor/LatentSync/checkpoints-v'+args.version.replace('.',''));folder.mkdir(exist_ok=True)
target=folder/'latentsync_unet.pt';part=folder/'parallel.partial';state=folder/'parallel.json'
size=m.size;block=8*1024*1024
if target.exists():
    digest=hashlib.sha256()
    with target.open('rb') as stream:
        while data:=stream.read(block):digest.update(data)
    if digest.hexdigest()==m.etag:print('Checkpoint already verified');raise SystemExit(0)
    raise RuntimeError('Existing checkpoint failed integrity verification; preserve it for diagnosis')
completed=set(json.loads(state.read_text())) if state.exists() else set()
fd=os.open(part,os.O_CREAT|os.O_RDWR,0o600);os.ftruncate(fd,size)
if not completed:
    existing=list((folder/'.cache/huggingface/download').glob('*.incomplete'))
    if existing:
        source=max(existing,key=lambda p:p.stat().st_size)
        with source.open('rb') as f:
            for index in range(source.stat().st_size//block):
                data=f.read(block)
                if len(data)!=block:break
                os.pwrite(fd,data,index*block);completed.add(index)
def download(index):
    start=index*block;end=min(size,start+block)-1
    for attempt in range(5):
        try:
            response=requests.get(u,headers={'Range':f'bytes={start}-{end}'},timeout=120)
            expected=f'bytes {start}-{end}/{size}'
            if response.status_code!=206 or response.headers.get('Content-Range')!=expected:raise RuntimeError('Server returned incorrect byte range')
            data=response.content
            if len(data)!=end-start+1:raise RuntimeError('Incomplete range')
            os.pwrite(fd,data,start);return index
        except Exception:
            if attempt==4:raise
            time.sleep(2**attempt)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for index in pool.map(download,[i for i in range((size+block-1)//block) if i not in completed]):
        completed.add(index);state.write_text(json.dumps(sorted(completed)))
        if len(completed)%10==0:print(f'{len(completed)*block/size:.0%} downloaded',flush=True)
os.close(fd)
h=hashlib.sha256()
with part.open('rb') as f:
    while data:=f.read(block):h.update(data)
if h.hexdigest()!=m.etag:raise RuntimeError('Checkpoint SHA-256 mismatch; do not load')
part.replace(target);print('Checkpoint verified',h.hexdigest(),flush=True)
