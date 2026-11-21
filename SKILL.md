---
name: video-to-td-corridor
description: Turn a forward-moving video (dolly / walk / drive down a street or hallway) into a 3D point-cloud scan and build the "3DGS infinite corridor" treadmill effect in a running TouchDesigner 2025 session with POPs. Use when asked to make an infinite corridor, endless street, splat/point-cloud treadmill, or to scan a video into TouchDesigner.
---

# Video → 3D scan → infinite corridor in TouchDesigner

The effect: a strip of scanned points slides past a **fixed** camera, wraps with
`mod`, and both ends dissolve into noise so the wrap seam is never seen. The scan
is repeated `TILES` times (default 3) end to end ahead of the camera, and the
camera sits at the near end, so the whole frame travels forward.

Sky: the render is composited over a heavily blurred frame of the video, and
distant points fade into a haze colour. Do **not** fill the distance with a sharp static still of the video: it covers most of
the frame and makes the POV look frozen even though the scan is moving (the
`Backdrop Brightness` parameter exists but defaults to 0 = off).

Scripts are in `scripts/`. Reference result: a 25 s street dolly, 151 frames,
13 million points, 60 fps at 1280x720 on an Apple M4 Pro.

## 1. Talk to TouchDesigner

TouchDesigner must already be open. Any way of running Python inside it works:

- **Textport** (Alt+T): paste the `exec(open(...).read())` line from step 3.
- **A TouchDesigner MCP / bridge** that exposes an "execute Python" tool.
  `scripts/td_bridge.py` is a small stdio JSON-RPC client for the TDMCP
  codex-bridge; adapt the server path to whichever bridge is installed:

```bash
python3 scripts/td_bridge.py -q execute_code '{}' code /path/to/snippet.py
```

- Bridge replies truncate long values. Have the snippet write JSON to a file and read that file.
- Set `result = ...` in the snippet for a short return value.
- The open project may hold other work. Only create/destroy your own component
  (`/project1/infinite_corridor`); never save the user's `.toe` without asking.
  Save your component with `comp.save('x.tox')` instead, and tell the user which
  `.toe` is open and where it lives, because their save goes to that location.
- Check what you built by saving a TOP to disk (`op('out1').save('x.jpg')`) and looking at it.

## 2. Make the scan

Needs a venv with `torch transformers pillow numpy opencv-python-headless`
(about 600 MB) and the model `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`
(1.2 GB, safetensors, cached in `~/.cache/huggingface`). Ask before downloading.
For interiors use the `Metric-Indoor` variant.

```bash
mkdir -p scan/frames && cp scripts/depth.py scripts/fuse.py scan/
ffmpeg -i video.mp4 -vf "select='not(mod(n\,4))',scale=1920:-1" -fps_mode passthrough -q:v 2 scan/frames/f_%03d.jpg
.venv/bin/python scan/depth.py             # about 1 s per frame on Apple silicon
.venv/bin/python scan/fuse.py 151 24 0.05  # frames, metres of depth kept per frame, voxel size
```

- **Use the whole shot, densely**: about 150 frames (every 4th frame of a 25 s
  clip). 25 frames left torn walls and holes. Drop frames that are not the shot
  (cuts, fades) with `lte(n\,LAST)` in the select filter.
- `fuse.py` finds the focal length by trying values and keeping the one with the
  most PnP inliers, poses every frame with SIFT + PnP against the keyframe before
  it (keyframes every N/25 frames, to limit drift), rescales each frame's depth,
  then **carves**: a point that two other frames see straight through is a ghost
  or smear and is dropped. Survivors are de-duplicated on a voxel grid keeping
  the closest observation.
- Two outputs, same coordinates (x right, y up, travel along −z, centred on
  z = 0, origin 1.7 m above the ground):
  - `street_scan.ply`: the detailed near layer (depth < kept value).
  - `street_far.ply`: building faces beyond that, up to 75 m, coarser (0.12 m),
    sky pixels removed, and removed wherever the near layer already has a
    surface. Without it tall buildings are cut off and the top of the frame is empty.
  - `street_scan.json`: `length`, `cam_x` (an obstacle-free lane for the camera),
    `focal_px`, `voxel`, `far_voxel`.
- Look at `scan/preview_top.jpg`, `preview_side.jpg` and `preview_far_front.jpg`
  before going on. Ground markings and signs should be readable.
- Tuning: far ground curling upward → lower the depth-kept value. Walls missing
  → raise `HALF_WIDTH` or the depth-kept value. Too many points → raise the voxel size.

## 3. Build the corridor

Run inside TouchDesigner:

```python
CORRIDOR_VIDEO = '/path/to/video.mp4'
CORRIDOR_SCAN = '/path/to/scan/street_scan.ply'   # optional
exec(open('/path/to/scripts/build_infinite_corridor.py').read())
```

It rebuilds `/project1/infinite_corridor` from scratch each time, so iterate by
editing the script and re-running. The first-person view is `out1`; an outside
view is `render_overview`.

Controls are custom parameters on the component (select it, press `p`):
`Travel Speed`, `Corridor Length`, `Edge Dispersion Zone`, noise and swirl
strength, point sizes, `Sky Brightness`, `Haze Colour`, `Distance Haze`, and a
**Record** page with a toggle that writes `out1` to a movie file.

If no scan file exists it falls back to "Video frame" mode: one frame projected
onto a ground + two walls box using the vanishing point. Quick, but far sections
are blurry and everything is flat.

## Things that cost time (TouchDesigner 2025.33230)

- **GLSL POP uniforms** default to type `float`. Set `vecNtype = 'vec4'` or
  `.y/.z/.w` fail with "vector swizzle selection out of range".
- **Sequence parameters** need their block count set first:
  `g.seq.vec.numBlocks = 5`, `g.seq.attr.numBlocks = 2`.
- **Output attributes**: list them in `outputattrs` (`'P Color PointScale'`).
  An attribute that does not exist on the input must be created in the
  Create Attributes sequence (`attr0name = 'color'`, `'pointscale'`). Creating
  one that already exists fails with "nameless block contains a member that
  already has a name".
- Reading a missing input attribute (`TDIn_Color()` on a grid) is a compile error.
  That is why the two sources have separate shaders.
- Compile errors are in the `<name>_info` DAT docked to the GLSL POP, not in `errors()`.
- Textures in a compute shader: bind on the Samplers page and use `textureLod(s, uv, 0.0)`.
- **Point File In POP**: PLY colours arrive as 0–255, so divide by 255. Set
  `colorfields = 'red green blue'` and clear `normalfields` to match the file.
- **Point Sprite MAT**: `sizingmodel = 'perspective'` and
  `pointscaleattrib = 'PointScale'`. For a scan with voxel size v, a scale near
  `v * 1.6 * 0.35` fills gaps without turning blocky.
- Grid POP needs `surftype = 'points'` to render as points.
- Non-commercial licence caps every TOP at 1280×1280; render at 1280×720.
- **Movie File Out TOP** records `out1`; the default codec writes large `.mov`
  files, so convert with `ffmpeg -i in.mov -c:v libx264 -crf 18 -pix_fmt yuv420p out.mp4`.
  One seamless loop lasts `Corridor Length / Travel Speed` seconds.
- Compute the treadmill phase in the parameter expression
  (`(absTime.seconds * speed) % length`) so the shader never sees a huge float.
- Put the camera in the scan's `cam_x` lane, not on the film camera's path, or
  the loop flies through walls where the film camera never went.

- Tiling: one GLSL POP per copy (`uTile.x` = copy index) into a Merge POP; the
  dissolve is computed over the whole tiled length, not per copy. Three copies of
  4.7M points still ran at 60 fps.

## Checking the result

The test that matters is "does the POV travel forward": in consecutive frames the
buildings ahead must grow and pass the camera across the whole frame, not only at
the edges.

Save `render1` at several moments a couple of seconds apart and look at all of
them: one good frame says nothing about the rest of the loop. Also save
`render_overview` to confirm both ends of the strip are dissolving.

## Known limits

- This is a depth-fused point cloud, not a trained Gaussian splat: surfaces are
  rougher than a real 3DGS scan, thin objects can smear into rows of copies where
  the estimated camera motion drifts, and the coarse far layer looks blocky when
  it passes close to the camera.
- A slow, straight dolly sees each surface from nearly one direction, so side
  walls are the weakest part. Footage with more parallax scans better.
- Metric scale from the depth model can be off by about 2x; only proportions matter here.
