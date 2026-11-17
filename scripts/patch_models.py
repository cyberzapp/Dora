"""Repair files absent in the pinned Mac port using the pinned upstream wrappers."""
from pathlib import Path
import shutil
root = Path(__file__).resolve().parents[1]
target = root / 'vendor/musetalk-mac/upstream/musetalk/models'
shutil.copytree(root / 'vendor/MuseTalk/musetalk/models', target, dirs_exist_ok=True)
p = target / 'unet.py'
s = p.read_text()
s = s.replace('weights = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path, map_location=self.device)', 'weights = torch.load(model_path, map_location="cpu", weights_only=True)')
p.write_text(s)

p = target / 'vae.py'
s = p.read_text().replace('init_latent_dist.sample()', 'init_latent_dist.mode()')
p.write_text(s)
