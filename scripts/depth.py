"""Metric depth for video frames with Depth Anything V2 (outdoor)."""
import sys, glob, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL = 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf'
here = os.path.dirname(os.path.abspath(__file__))
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
proc = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(dev).eval()
files = sorted(glob.glob(os.path.join(here, 'frames', 'f_*.jpg')))
if len(sys.argv) > 1:
    files = files[:int(sys.argv[1])]
os.makedirs(os.path.join(here, 'depth'), exist_ok=True)
for f in files:
    img = Image.open(f).convert('RGB')
    with torch.no_grad():
        out = model(**proc(images=img, return_tensors='pt').to(dev))
    d = proc.post_process_depth_estimation(out, target_sizes=[(img.height, img.width)])[0]['predicted_depth']
    d = d.float().cpu().numpy()
    np.save(os.path.join(here, 'depth', os.path.basename(f)[:-4] + '.npy'), d.astype(np.float16))
    v = np.clip(d / 80.0, 0, 1) ** 0.5
    Image.fromarray((255 * (1 - v)).astype(np.uint8)).save(os.path.join(here, 'depth', os.path.basename(f)[:-4] + '.png'))
    print(os.path.basename(f), 'depth m: min %.1f med %.1f max %.1f' % (d.min(), np.median(d), d.max()), flush=True)
