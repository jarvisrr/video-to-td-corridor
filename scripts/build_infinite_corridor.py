"""
Infinite corridor (TouchDesigner 2025 POPs) built from a forward-moving video.

Source "Depth scan": a coloured point cloud fused from the video's frames
(depth.py + fuse.py) is treadmilled past a fixed camera with a modulo wrap. The
scan is repeated TILES times ahead of the camera, both ends dissolve into noise
so the loop seam is never visible, and a blurred frame of the video is the sky.
Source "Video frame": no scan needed - one frame is projected onto a simple
ground + two walls street shape.

Run in the TouchDesigner Textport (or through any bridge that executes Python in TD):
    CORRIDOR_VIDEO = '/path/to/video.mp4'
    CORRIDOR_SCAN = '/path/to/scan/street_scan.ply'      # optional, defaults to scan/ next to the video
    exec(open('/path/to/scripts/build_infinite_corridor.py').read())

It rebuilds /project1/infinite_corridor from scratch each time.
"""

import os, json
VIDEO = globals().get('CORRIDOR_VIDEO', '')
if not VIDEO:
    raise ValueError('Set CORRIDOR_VIDEO to the video file path before running this script')
SCAN = globals().get('CORRIDOR_SCAN', os.path.join(os.path.dirname(VIDEO), 'scan', 'street_scan.ply'))
HAS_SCAN = os.path.exists(SCAN)
FARSCAN = os.path.join(os.path.dirname(SCAN), 'street_far.ply')   # tall / distant building faces, coarser
HAS_FAR = HAS_SCAN and os.path.exists(FARSCAN)
META = json.load(open(SCAN[:-4] + '.json')) if HAS_SCAN else {}
TILES = int(globals().get('CORRIDOR_TILES', 3))   # copies of the scan laid end to end ahead of the camera
COLS, ROWS = 1400, 700           # points around the street section x along it (980k)

SHADER = r"""
// uTime  = (phase, unused, corridorLength, edgeZone)
// uNoise = (noiseStrength, noiseScale, swirl, pointSizeMul)
// uCam   = (vanishX, vanishY, focal, aspect)      vanish in -0.5..0.5, y up
// uBox   = (halfWidth, camHeight, wallHeight, mode)   mode 0 = video, 1 = ply
// uRes   = (cols, rows, time, jitter)
// sVideo = the video frame

vec3 hash33(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}
float noise3d(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(dot(hash33(i), f),
                       dot(hash33(i + vec3(1,0,0)), f - vec3(1,0,0)), u.x),
                   mix(dot(hash33(i + vec3(0,1,0)), f - vec3(0,1,0)),
                       dot(hash33(i + vec3(1,1,0)), f - vec3(1,1,0)), u.x), u.y),
               mix(mix(dot(hash33(i + vec3(0,0,1)), f - vec3(0,0,1)),
                       dot(hash33(i + vec3(1,0,1)), f - vec3(1,0,1)), u.x),
                   mix(dot(hash33(i + vec3(0,1,1)), f - vec3(0,1,1)),
                       dot(hash33(i + vec3(1,1,1)), f - vec3(1,1,1)), u.x), u.y), u.z);
}
vec3 noiseVec(vec3 p) {
    return vec3(noise3d(p + vec3(13.1, 0.0, 0.0)),
                noise3d(p + vec3(0.0, 29.7, 0.0)),
                noise3d(p + vec3(0.0, 0.0, 47.3)));
}

void main() {
    const uint id = TDIndex();
    if (id >= TDNumElements()) return;

    float L = max(uTime.z, 0.1);
    float halfL = 0.5 * L;
    vec3 pos;
    vec4 col = vec4(1.0);
    float size = 0.02;
    float vis = 1.0;

    if (uBox.w < 0.5) {
        // --- lay the point on the street-canyon surfaces (ground + 2 walls) ---
        uint cols = uint(uRes.x), rows = uint(uRes.y);
        vec3 jit = hash33(vec3(float(id % cols), float(id / cols), 7.0)) * uRes.w;
        float W = uBox.x, h = uBox.y, H = uBox.z;
        float wallLen = h + H;
        float perim = 2.0 * W + 2.0 * wallLen;
        float s = (float(id % cols) + 0.5 + jit.x) / float(cols) * perim;
        uint capRows = rows / 8u, row = id / cols;
        bool cap = row >= rows - capRows;      // last rows: static backdrop closing the far end
        float d = (float(row) + 0.5 + jit.y) / float(rows - capRows) * L;   // distance ahead of the film camera

        vec3 p;
        if (s < wallLen)                  p = vec3(-W, H - s, -d);                 // left wall, top to bottom
        else if (s < wallLen + 2.0 * W)   p = vec3(-W + (s - wallLen), -h, -d);    // ground
        else                              p = vec3( W, -h + (s - wallLen - 2.0 * W), -d);   // right wall
        if (cap) {
            vec2 t = vec2((float(id % cols) + 0.5) / float(cols), (float(row - (rows - capRows)) + 0.5) / float(capRows));
            p = vec3(mix(-W, W, t.x) * 2.0, mix(-h - 1.0, H * 1.5, t.y), -halfL);   // oversized so the dissolving walls never expose black
        }

        // --- project back into the video frame for colour ---
        float f = uCam.z;
        vec3 fwd = normalize(vec3(uCam.x * uCam.w, uCam.y, -f));   // corridor axis in film-camera space
        vec3 zw = -fwd;
        vec3 xw = normalize(cross(vec3(0.0, 1.0, 0.0), zw));
        vec3 yw = cross(zw, xw);
        vec3 c = xw * p.x + yw * p.y + zw * p.z;
        vec2 uv = vec2(c.x / uCam.w, c.y) * (f / max(-c.z, 1e-4));
        vec2 b = 0.5 - abs(uv);                                    // distance to frame border
        vis = (c.z < 0.0) ? smoothstep(0.0, 0.03, min(b.x, b.y)) : 0.0;
        col = textureLod(sVideo, uv + 0.5, 0.0);

        pos = p + vec3(0.0, 0.0, halfL);
        size = max(perim / float(cols), L / float(rows - capRows)) * (1.0 + 0.5 * jit.z);
        if (cap) {
            P[id] = p;
            Color[id] = vec4(col.rgb, 1.0);
            PointScale[id] = max(4.0 * W / float(cols), (wallLen * 1.5 + 1.0) / float(capRows)) * 1.5 * vis;
            return;
        }
    } else {
        pos = TDIn_P();      // point-file mode: positions only, drawn white
    }

    // --- treadmill: slide along the corridor axis and wrap ---
    float z = mod(pos.z + uTime.x + halfL, L) - halfL;
    pos.z = z;

    // --- dissolve both ends so the wrap seam is never seen ---
    float nd = abs(z) / halfL;                         // 0 centre, 1 seam
    float edge = smoothstep(1.0 - clamp(uTime.w, 0.01, 0.95), 1.0, nd);
    if (edge > 0.0001) {
        vec3 n = noiseVec(pos * uNoise.y + vec3(0.0, 0.0, uRes.z * 0.35));
        vec2 radial = length(pos.xy) > 0.001 ? normalize(pos.xy) : vec2(0.0, 1.0);
        pos += n * (uNoise.x * edge);
        pos.xy += radial * (uNoise.z * edge * edge);
        float a = edge * edge * 0.6 * sign(z);         // twist so the ends curl like dust
        pos.xy = mat2(cos(a), -sin(a), sin(a), cos(a)) * pos.xy;
    }
    float fade = 1.0 - smoothstep(0.85, 1.0, nd);

    col.rgb *= mix(0.3, 1.0, fade);
    col.a = 1.0;

    P[id] = pos;
    Color[id] = col;
    PointScale[id] = size * uNoise.w * mix(1.0, 0.45, edge) * fade * vis;
}
""".strip()


SHADER_SCAN = SHADER.split('void main()')[0] + r"""
// scan mode: uBox = (voxelSize, colourScale, extra point size (far layer), 1)
// uHaze = (r, g, b, unused) colour distant points fade into
// uTile = (copy index, copies, camera position inside the near dissolve zone 0..1, fog)
// The strip is repeated `copies` times ahead of the camera (which sits at z = 0
// looking down -z), so the whole view travels; points appear out of dust at the
// far end and fly apart as they pass the camera.
void main() {
    const uint id = TDIndex();
    if (id >= TDNumElements()) return;

    float L = max(uTime.z, 0.1);
    float halfL = 0.5 * L;
    float total = L * max(uTile.y, 1.0);
    float zone = clamp(uTime.w, 0.01, 0.95) * L;      // length of each dissolve zone
    float B = zone * uTile.z;                         // how far the strip reaches behind the camera
    vec3 pos = TDIn_P();
    vec4 col = TDIn_Color() * uBox.y;
    vec3 jit = hash33(pos * 37.0);
    float size = uBox.x * (1.6 + 0.5 * jit.z) * max(uBox.z, 1.0);

    float zt = mod(pos.z + halfL + uTime.x + uTile.x * L, total);   // 0 far end .. total near end
    float z = zt - (total - B);
    pos.z = z;

    float dn = (B - z) / zone;                        // 0 at the near end
    float df = zt / zone;                             // 0 at the far end
    float eN = 1.0 - smoothstep(0.0, 1.0, dn);
    float eF = 1.0 - smoothstep(0.0, 1.0, df);
    float edge = max(eN, eF);
    if (edge > 0.0001) {
        vec3 n = noiseVec(pos * uNoise.y + vec3(0.0, 0.0, uRes.z * 0.35));
        vec2 radial = length(pos.xy) > 0.001 ? normalize(pos.xy) : vec2(0.0, 1.0);
        pos += n * (uNoise.x * edge);
        pos.xy += radial * (uNoise.z * edge * edge);
        float a = edge * edge * 0.6 * (eN > eF ? 1.0 : -1.0);
        pos.xy = mat2(cos(a), -sin(a), sin(a), cos(a)) * pos.xy;
    }
    float fade = smoothstep(0.0, 0.15, min(dn, df));
    float fog = uTile.w * smoothstep(0.2, 1.0, -z / (total - B));   // haze with distance
    col.rgb = mix(col.rgb, uHaze.rgb, fog) * mix(0.5, 1.0, fade);
    col.a = 1.0;

    P[id] = pos;
    Color[id] = col;
    PointScale[id] = size * uNoise.w * mix(1.0, 0.45, edge) * fade;
}
"""


def _set(o, name, value=None, expr=None):
    p = getattr(o.par, name, None)
    if p is None:
        print('  !! missing par', o.name, name)
        return False
    if expr is not None:
        p.expr = expr
        p.mode = ParMode.EXPRESSION
    else:
        p.val = value
    return True


def _float(page, name, label, val, lo, hi):
    p = page.appendFloat(name, label=label)[0]
    p.default = val; p.val = val
    p.normMin = lo; p.normMax = hi
    return p


def build_infinite_corridor(parent_path='/project1'):
    parent = op(parent_path)
    old = parent.op('infinite_corridor')
    pos = (old.nodeX, old.nodeY) if old else (0, -600)
    if old:
        old.destroy()

    root = parent.create(baseCOMP, 'infinite_corridor')
    root.nodeX, root.nodeY = pos

    # ---------------- controls ----------------
    pg = root.appendCustomPage('Corridor')
    src = pg.appendMenu('Source', label='Source')[0]
    src.menuNames = ['video', 'ply']; src.menuLabels = ['Video frame (flat walls)', 'Depth scan (.ply)']
    src.val = 'ply' if HAS_SCAN else 'video'
    f = pg.appendFile('Videofile', label='Video File')[0]; f.val = VIDEO
    _float(pg, 'Frame', 'Video Frame', 24, 0, 640)
    pf = pg.appendFile('Plyfile', label='Scan .ply File')[0]; pf.val = SCAN if HAS_SCAN else ''
    _float(pg, 'Scanvoxel', 'Scan Point Spacing', META.get('voxel', 0.06), 0.01, 0.3)
    _float(pg, 'Speed', 'Travel Speed', 6.0 if HAS_SCAN else 1.2, -30, 30)
    _float(pg, 'Corridorlength', 'Corridor Length', round(META.get('length', 20), 2), 4, 80)
    _float(pg, 'Edgefade', 'Edge Dispersion Zone', 0.35, 0.02, 0.95)
    _float(pg, 'Noisestrength', 'Noise Strength', 3.0 if HAS_SCAN else 1.6, 0, 10)
    _float(pg, 'Noisescale', 'Noise Scale', 0.3 if HAS_SCAN else 0.6, 0.05, 5)
    _float(pg, 'Swirlstrength', 'Outward Swirl', 4.0 if HAS_SCAN else 2.0, 0, 10)
    _float(pg, 'Pointsize', 'Point Size', 0.35 if HAS_SCAN else 1.1, 0.05, 6)
    _float(pg, 'Jitter', 'Point Jitter', 0.3, 0, 2)
    _float(pg, 'Scancolor', 'Scan Colour Scale', 1.0, 0, 2)
    _float(pg, 'Backdrop', 'Backdrop Brightness', 0.0, 0, 1)     # static still of the video; 0 = off (it makes the view look frozen)
    _float(pg, 'Campos', 'Camera In Near Dissolve', 0.7, 0, 1)
    _float(pg, 'Fog', 'Distance Haze', 0.75, 0, 1)
    ff = pg.appendFile('Farfile', label='Far Layer .ply File')[0]; ff.val = FARSCAN if HAS_FAR else ''
    _float(pg, 'Farvoxel', 'Far Layer Point Spacing', META.get('far_voxel', 0.22), 0.05, 1)
    _float(pg, 'Farsize', 'Far Layer Point Size', 1.3, 0.5, 8)
    _float(pg, 'Sky', 'Sky Brightness', 0.85, 0, 1.5)
    hz = pg.appendRGB('Haze', label='Haze Colour')
    for hp, hv in zip(hz, (0.80, 0.74, 0.72)): hp.default = hv; hp.val = hv

    pg2 = root.appendCustomPage('Unproject')
    _float(pg2, 'Vanishx', 'Vanishing Point X', -0.01, -0.5, 0.5)
    _float(pg2, 'Vanishy', 'Vanishing Point Y', -0.14, -0.5, 0.5)
    _float(pg2, 'Focal', 'Focal Length', 0.55, 0.2, 2)
    _float(pg2, 'Halfwidth', 'Street Half Width', 2.4, 0.5, 10)
    _float(pg2, 'Camheight', 'Camera Height', 1.0, 0.2, 5)
    _float(pg2, 'Ceilheight', 'Wall Height', 6.0, 1, 30)

    # ---------------- video source ----------------
    mov = root.create(moviefileinTOP, 'video_in')
    mov.nodeX, mov.nodeY = -600, 0
    _set(mov, 'file', expr="parent().par.Videofile")
    _set(mov, 'playmode', 'specify')
    _set(mov, 'indexunit', 'frames')
    _set(mov, 'index', expr="parent().par.Frame")

    fit = root.create(resolutionTOP, 'video_res')
    fit.nodeX, fit.nodeY = -420, 0
    _set(fit, 'outputresolution', 'custom')
    _set(fit, 'resolutionw', 1280)
    _set(fit, 'resolutionh', 720)
    fit.inputConnectors[0].connect(mov)

    # ---------------- geometry ----------------
    geo = root.create(geometryCOMP, 'geo_splats')
    geo.nodeX, geo.nodeY = -200, 0
    for c in list(geo.children):
        c.destroy()

    t2p = geo.create(gridPOP, 'corridor_points')
    t2p.nodeX, t2p.nodeY = -400, -120
    _set(t2p, 'surftype', 'points')
    _set(t2p, 'cols', COLS)
    _set(t2p, 'rows', ROWS)

    ply = geo.create(pointfileinPOP, 'scan_in')
    ply.nodeX, ply.nodeY = -400, 120
    _set(ply, 'file', expr="parent(2).par.Plyfile")
    _set(ply, 'colorfields', 'red green blue')
    _set(ply, 'normalfields', '')

    R = "parent(2).par."

    def wrap(name, source, shader, y, box, new_attrs, tile=0):
        dat = geo.create(textDAT, name + '_shader')
        dat.nodeX, dat.nodeY = -200, y + 110
        dat.text = shader
        g = geo.create(glslPOP, name)
        g.nodeX, g.nodeY = -200, y
        g.inputConnectors[0].connect(source)
        _set(g, 'computedat', dat.name)
        _set(g, 'outputattrs', 'P Color PointScale')
        g.seq.attr.numBlocks = len(new_attrs)      # attributes the shader creates
        for i, a in enumerate(new_attrs):
            _set(g, f'attr{i}name', a)
        vecs = [
            ('uTime',  [f"(absTime.seconds * {R}Speed) % {R}Corridorlength", "0",
                        R + "Corridorlength", R + "Edgefade"]),
            ('uNoise', [R + "Noisestrength", R + "Noisescale", R + "Swirlstrength", R + "Pointsize"]),
            ('uCam',   [R + "Vanishx", R + "Vanishy", R + "Focal",
                        "op('../video_res').width / op('../video_res').height"]),
            ('uBox',   box),
            ('uRes',   [str(COLS), str(ROWS), "absTime.seconds % 1000", R + "Jitter"]),
            ('uTile',  [str(tile), str(TILES), R + "Campos", R + "Fog"]),
            ('uHaze',  [R + "Hazer * " + R + "Sky", R + "Hazeg * " + R + "Sky", R + "Hazeb * " + R + "Sky", "0"]),
        ]
        g.seq.vec.numBlocks = len(vecs)
        for i, (vn, exprs) in enumerate(vecs):
            _set(g, f'vec{i}name', vn)
            _set(g, f'vec{i}type', 'vec4')
            for comp, e in zip('xyzw', exprs):
                _set(g, f'vec{i}value{comp}', expr=e)
        _set(g, 'sampler0name', 'sVideo')
        _set(g, 'sampler0top', '../video_res')
        return g

    g_video = wrap('corridor_wrap_video', t2p, SHADER, -120,
                   [R + "Halfwidth", R + "Camheight", R + "Ceilheight", "0"], ['color', 'pointscale'])
    scans = [wrap('corridor_wrap_scan' + ('' if k == 0 else str(k)), ply, SHADER_SCAN, 120 + 240 * k,
                  [R + "Scanvoxel", R + "Scancolor / 255", "0", "1"], ['pointscale'], k)   # the scan already has Color
             for k in range(TILES)]
    if HAS_FAR:
        far = geo.create(pointfileinPOP, 'far_in')
        far.nodeX, far.nodeY = -400, 120 + 240 * TILES
        _set(far, 'file', expr="parent(2).par.Farfile")
        _set(far, 'colorfields', 'red green blue')
        _set(far, 'normalfields', '')
        scans += [wrap('corridor_wrap_far' + str(k), far, SHADER_SCAN, 120 + 240 * (TILES + k),
                       [R + "Farvoxel", R + "Scancolor / 255", R + "Farsize", "1"], ['pointscale'], k)
                  for k in range(TILES)]
    g_scan = scans[0]
    if len(scans) > 1:
        g_scan = geo.create(mergePOP, 'scan_tiles')
        g_scan.nodeX, g_scan.nodeY = -60, 240
        for k, sc in enumerate(scans):
            g_scan.inputConnectors[k].connect(sc)

    sw = geo.create(switchPOP, 'source_switch')
    sw.nodeX, sw.nodeY = 0, 0
    sw.inputConnectors[0].connect(g_video)
    sw.inputConnectors[1].connect(g_scan)
    _set(sw, 'index', expr="parent(2).par.Source.menuIndex")
    g = sw

    out = geo.create(nullPOP, 'OUT_SPLATS')
    out.nodeX, out.nodeY = 200, 0
    out.inputConnectors[0].connect(g)
    out.display = True
    out.render = True

    mat = root.create(pointspriteMAT, 'splat_mat')
    mat.nodeX, mat.nodeY = -200, -160
    _set(mat, 'pointscaleattrib', 'PointScale')
    _set(mat, 'sizingmodel', 'perspective')
    _set(geo, 'material', mat.name)

    # ---------------- camera / render ----------------
    # travel down the lane the scan reports as free of obstacles
    CAMX = META.get('cam_x', 0) if HAS_SCAN else 0
    cam = root.create(cameraCOMP, 'cam1')
    cam.nodeX, cam.nodeY = -200, 160
    for n, v in (('tx', CAMX), ('ty', 0.5 if HAS_SCAN else 0), ('tz', 0), ('rx', 10 if HAS_SCAN else 6), ('fov', 80), ('near', 0.05), ('far', 600)):
        _set(cam, n, v)

    # ---------------- distant backdrop (scan mode) ----------------
    # The scan only holds what was near the camera; a dimmed still of the video,
    # placed past the far dissolve, fills the distance instead of black.
    if HAS_SCAN:
        D = META['length'] * TILES + 6.0                  # just beyond the far end
        fl = META['focal_px'] / 1080.0                    # focal length in frame heights
        bh = D / fl; bw = bh * 16.0 / 9.0
        bg = root.create(geometryCOMP, 'geo_backdrop')
        bg.nodeX, bg.nodeY = -200, -320
        for c in list(bg.children):
            c.destroy()
        rect = bg.create(rectangleSOP, 'rect')
        _set(rect, 'sizex', bw); _set(rect, 'sizey', bh)
        rect.display = True; rect.render = True
        bmat = root.create(constantMAT, 'backdrop_mat')
        bmat.nodeX, bmat.nodeY = -400, -320
        _set(bmat, 'colormap', 'video_res')
        for c in 'rgb':
            _set(bmat, 'color' + c, expr="parent().par.Backdrop")
        _set(bg, 'material', 'backdrop_mat')
        _set(bg, 'tx', CAMX)
        _set(bg, 'ty', 0.5 - root.par.Vanishy.eval() * bh)    # vanishing point at eye level
        _set(bg, 'tz', -D)
        _set(bg, 'render', expr="parent().par.Source.menuIndex == 1 and parent().par.Backdrop > 0")

    render = root.create(renderTOP, 'render1')
    render.nodeX, render.nodeY = 40, 0
    _set(render, 'camera', 'cam1')
    _set(render, 'geometry', 'geo_splats geo_backdrop' if HAS_SCAN else 'geo_splats')
    _set(render, 'outputresolution', 'custom')
    _set(render, 'resolutionw', 1280)
    _set(render, 'resolutionh', 720)
    _set(render, 'bgcolora', 1)

    # top-down overview like the lower viewport of the reference
    OV = 2.2 if HAS_SCAN else 1.0
    cam2 = root.create(cameraCOMP, 'cam_overview')
    cam2.nodeX, cam2.nodeY = -200, 300
    OVZ = -META['length'] * (TILES - 1) * 0.5 if HAS_SCAN else 0     # centre of the tiled strip
    if HAS_SCAN: OV *= 0.6 + 0.4 * TILES
    if HAS_FAR: OV *= 1.7                                            # tall building faces need a wider view
    for n, v in (('tx', 20 * OV), ('ty', 24 * OV), ('tz', 15 * OV + OVZ), ('rx', -45), ('ry', 53), ('fov', 50), ('far', 900)):
        _set(cam2, n, v)
    over = root.create(renderTOP, 'render_overview')
    over.nodeX, over.nodeY = 40, 160
    _set(over, 'camera', 'cam_overview')
    _set(over, 'geometry', 'geo_splats')
    _set(over, 'outputresolution', 'custom')
    _set(over, 'resolutionw', 1280)
    _set(over, 'resolutionh', 720)
    _set(over, 'bgcolora', 1)

    # sky: a heavily blurred frame of the video behind the points. It holds no
    # recognisable objects, so nothing in the view looks frozen.
    final = render
    if HAS_SCAN:
        _set(render, 'bgcolora', 0)
        blur = root.create(blurTOP, 'sky_blur')
        blur.nodeX, blur.nodeY = -420, -160
        blur.inputConnectors[0].connect(fit)
        _set(blur, 'size', 120)
        sky = root.create(levelTOP, 'sky')
        sky.nodeX, sky.nodeY = -300, -160
        sky.inputConnectors[0].connect(blur)
        _set(sky, 'brightness1', expr="parent().par.Sky")
        final = root.create(overTOP, 'with_sky')
        final.nodeX, final.nodeY = 140, 0
        final.inputConnectors[0].connect(render)
        final.inputConnectors[1].connect(sky)

    o = root.create(outTOP, 'out1')
    o.nodeX, o.nodeY = 240, 0
    o.inputConnectors[0].connect(final)
    o.viewer = True

    # ---------------- recording ----------------
    # Toggle "Record" on the component to write the output to a movie file.
    pg3 = root.appendCustomPage('Record')
    rf = pg3.appendFile('Recordfile', label='Movie File')[0]
    rf.val = os.path.join(os.path.dirname(VIDEO), 'corridor_output.mov')
    pg3.appendToggle('Record', label='Record')
    rec = root.create(moviefileoutTOP, 'record')
    rec.nodeX, rec.nodeY = 240, -160
    rec.inputConnectors[0].connect(final)
    _set(rec, 'file', expr="parent().par.Recordfile")
    _set(rec, 'record', expr="parent().par.Record")

    print('Built', root.path)
    return root


build_infinite_corridor('/project1')
