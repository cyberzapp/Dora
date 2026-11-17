"""Local-only FlashHead diagnostics; installed weights are not a quality pass."""
import json
from pathlib import Path
import sys

from worker import ffmpeg
from flashhead_cache import sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    report=dict(engine='FlashHead Pro',local_only=True,visual_quality_approved=False,missing_or_damaged=[])
    try:
        import torch
        report['torch']=torch.__version__;report['mps_available']=torch.backends.mps.is_available()
        if report['mps_available']:
            values=torch.ones((8,8),device='mps');result=values@values;torch.mps.synchronize()
            report['metal_tensor_test']=bool(result[0,0].item()==8)
        for name,folder in [('flashhead-pro','flashhead-models'),('flashhead-audio','flashhead-audio')]:
            manifest=json.loads((ROOT/'backend/model_manifests'/f'{name}.json').read_text())
            for item in manifest['files']:
                path=ROOT/'.local'/folder/item['path']
                if not path.is_file() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
                    report['missing_or_damaged'].append(f'{folder}/{item["path"]}')
        report['ffmpeg']=ffmpeg()
        evidence=ROOT/'.local/flashhead-engine-recovery/report.json'
        if evidence.exists():
            recovery=json.loads(evidence.read_text())
            report['pretrained_window_recovery_verified']=recovery.get('resumed_window_matches_uninterrupted_exactly') is True
        report['installation_ready']=bool(report['mps_available'] and report.get('metal_tensor_test') and not report['missing_or_damaged'])
    except Exception as error:report['error']=str(error);report['installation_ready']=False
    print(json.dumps(report,indent=2))
    return 0 if report['installation_ready'] else 1


if __name__=='__main__':sys.exit(main())
