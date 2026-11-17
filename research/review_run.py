"""Finish reproducible local quality checks after a trained render completes."""
import argparse,json,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',required=True)
    parser.add_argument('--wait-seconds',type=int,default=0)
    args=parser.parse_args();folder=Path(args.benchmark).resolve()
    start=time.monotonic()
    while not (folder/'report.json').exists():
        if time.monotonic()-start>=args.wait_seconds:raise TimeoutError('Completed render report unavailable')
        time.sleep(5)
    report=json.loads((folder/'report.json').read_text())
    progress=json.loads((folder/'progress.json').read_text())
    if progress['status']!='complete':raise RuntimeError('Render is not complete')
    def run(script,*arguments):
        print('Running '+script,flush=True)
        subprocess.run([sys.executable,str(ROOT/'research'/script),*[str(value) for value in arguments]],check=True,cwd=ROOT)
    run('recompose_candidate.py','--benchmark',folder,'--method','upstream')
    run('recompose_candidate.py','--benchmark',folder,'--appearance-video',report['source_video'])
    candidates=['upstream-single-encode.mov','masked-composite-original-source.mov']
    for candidate in candidates:
        name=Path(candidate).stem
        run('review_candidate.py','--candidate',folder/candidate,'--motion',report['source_video'],
            '--audio',report['source_audio'],'--seconds',report['seconds'],'--candidate-label',name,'--output',folder/name)
        run('temporal_review.py','--video',folder/candidate,'--output',folder/name)
        run('syncnet_timing.py','--video',folder/candidate,'--seconds',report['seconds'],'--output',folder/name/'syncnet.json')
    summary={'candidates':candidates,'checks_complete':True,'human_review_required':True,'quality_passed':False}
    (folder/'review-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
