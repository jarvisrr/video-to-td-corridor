# video-to-td-corridor

A Claude Code skill that turns a forward-moving video (a dolly, walk or drive
down a street or hallway) into a 3D point-cloud scan and builds the
"infinite corridor" effect in TouchDesigner 2025 with POPs: the scanned scene
streams past the camera forever, appearing out of dust in the distance and
flying apart as it passes.

Inspired by keita.kuki's "3dgs infinite corridor" (`#touchdesigner #tdpops`).

## How it works

1. **Frames** — ffmpeg pulls about 150 frames from the whole shot.
2. **Depth** — Depth Anything V2 (metric) estimates a depth map for every frame.
3. **Camera motion** — SIFT feature matches plus PnP against the depth give each
   frame's camera position; the focal length is found by trying values and
   keeping the one that explains the motion best.
4. **Fusion** — every frame's pixels are pushed into 3D. Points that other
   frames can see straight through (ghosts, smears) are carved away, and the
   rest are merged on a voxel grid. The result is two coloured `.ply` files: a
   detailed near layer and a coarser layer of tall / distant building faces.
5. **Corridor** — in TouchDesigner, a GLSL compute shader slides the points
   toward a fixed camera and wraps them with `mod`. The scan is repeated three
   times end to end so there is scene all the way into the distance, both ends
   dissolve into noise to hide the wrap, far points fade into haze, and a
   blurred frame of the video is the sky.

Everything is one generated component, `/project1/infinite_corridor`, with
parameters for speed, dissolve, point size, sky, haze and recording.

## Requirements

- TouchDesigner 2025 (POPs). The non-commercial licence works; output is capped at 1280x720.
- Python 3.9+ with the packages in `requirements.txt` (about 600 MB), and ffmpeg.
- The depth model `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`
  (1.2 GB, downloaded from Hugging Face on first run). Use the `Metric-Indoor`
  variant for interiors.
- Developed on Apple silicon (MPS); falls back to CPU elsewhere.

## Install as a skill

Copy this folder to `.claude/skills/video-to-td-corridor/` in a project, or to
`~/.claude/skills/video-to-td-corridor/` for all projects, then ask Claude for
an infinite corridor from a video.

## Use by hand

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p scan/frames && cp scripts/depth.py scripts/fuse.py scan/
ffmpeg -i video.mp4 -vf "select='not(mod(n\,4))',scale=1920:-1" -fps_mode passthrough -q:v 2 scan/frames/f_%03d.jpg
.venv/bin/python scan/depth.py
.venv/bin/python scan/fuse.py 151 24 0.05   # frame count, metres of depth kept per frame, voxel size
```

Then in the TouchDesigner Textport:

```python
CORRIDOR_VIDEO = '/path/to/video.mp4'
exec(open('/path/to/scripts/build_infinite_corridor.py').read())
```

View `out1` inside `/project1/infinite_corridor`.

## Files

| File | Purpose |
| --- | --- |
| `SKILL.md` | Instructions Claude follows, including the TouchDesigner pitfalls |
| `scripts/depth.py` | Depth maps for the extracted frames |
| `scripts/fuse.py` | Camera poses, carving, fusion, `.ply` export |
| `scripts/build_infinite_corridor.py` | Builds the TouchDesigner network |
| `scripts/td_bridge.py` | Optional helper to run Python in TouchDesigner through an MCP bridge |

## Limits

This is a point cloud from estimated depth, not a trained Gaussian splat.
Surfaces are rougher than a real 3DGS scan, thin objects can smear where the
estimated camera motion drifts, and a slow straight dolly gives weak coverage of
side walls. Footage with more sideways movement scans better.
