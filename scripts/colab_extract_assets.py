import zipfile
from pathlib import Path
z = Path('/content/colab_bundle.zip')
out = Path('/content/persona_assets')
out.mkdir(exist_ok=True)
with zipfile.ZipFile(z) as zf:
    zf.extractall(out)
roots = sorted({str(p.parent) for p in out.rglob('*_persona_vectors.pt')})
print('extracted roots', roots)
print('files', sorted(p.name for p in out.rglob('*') if p.is_file()))
import torch
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
