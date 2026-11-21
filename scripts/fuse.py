"""Fuse per-frame metric depth into one street point cloud (PLY).

Camera motion between consecutive frames comes from SIFT matches + PnP against
the previous frame's depth; each frame then contributes its nearby, high
resolution points, de-duplicated on a voxel grid.
"""
import os, glob, json, sys
import numpy as np, cv2

here = os.path.dirname(os.path.abspath(__file__))
NF = int(sys.argv[1]) if len(sys.argv) > 1 else 25
KEEP = float(sys.argv[2]) if len(sys.argv) > 2 else 28.0     # metres of depth kept per frame
VOX = float(sys.argv[3]) if len(sys.argv) > 3 else 0.06      # voxel size, metres
STRIDE = 1 if NF <= 40 else 2     # pixel step when unprojecting; dense frame sets overlap heavily
KS = max(1, round(NF / 25))      # keyframe spacing: poses are chained keyframe to keyframe to limit drift
HALF_WIDTH = 22.0     # metres kept either side of the street centre
EYE = 1.7             # origin sits this far above the ground

files = sorted(glob.glob(os.path.join(here, 'frames', 'f_*.jpg')))[:NF]
imgs = [cv2.imread(f) for f in files]
deps = [np.load(os.path.join(here, 'depth', os.path.basename(f)[:-4] + '.npy')).astype(np.float32) for f in files]
H, W = deps[0].shape
sift = cv2.SIFT_create(6000)
feats = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None) for im in imgs]
bf = cv2.BFMatcher()


def matches(i, j):
    (ka, da), (kb, db) = feats[i], feats[j]
    m = bf.knnMatch(da, db, k=2)
    good = [a for a, b in m if a.distance < 0.7 * b.distance]
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    return pa, pb


def Kmat(f):
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float64)


def pnp(i, j, f, scale_i=1.0):
    pa, pb = matches(i, j)
    d = deps[i][pa[:, 1].astype(int), pa[:, 0].astype(int)] * scale_i
    ok = (d > 0.5) & (d < 70)
    pa, pb, d = pa[ok], pb[ok], d[ok]
    X = np.stack([(pa[:, 0] - W / 2) / f * d, (pa[:, 1] - H / 2) / f * d, d], 1).astype(np.float64)
    okp, rvec, tvec, inl = cv2.solvePnPRansac(X, pb.astype(np.float64), Kmat(f), None,
                                              reprojectionError=3.0, iterationsCount=2000, confidence=0.999)
    if not okp or inl is None:
        return None
    inl = inl[:, 0]
    R, _ = cv2.Rodrigues(rvec)
    Xj = X[inl] @ R.T + tvec[:, 0]
    dj = deps[j][pb[inl, 1].astype(int), pb[inl, 0].astype(int)]
    s = float(np.median(Xj[:, 2] / np.maximum(dj, 1e-3)))     # depth scale that makes frame j agree
    return R, tvec[:, 0], len(inl), len(X), s


# ---- focal length: the one that explains the motion best ----
best = None
for fr in np.arange(0.40, 1.25, 0.05):
    f = fr * H
    score = 0
    for i in (0, NF // 3, 2 * NF // 3):
        r = pnp(i, i + KS, f)
        score += r[2] / max(r[3], 1) if r else 0
    print('focal %.2f*H inlier ratio %.3f' % (fr, score / 3), flush=True)
    if best is None or score > best[0]:
        best = (score, f)
f = best[1]
print('focal chosen: %.1f px (%.2f * H)' % (f, f / H))

# ---- chain camera poses (each frame against the keyframe before it) ----
T = [np.eye(4)]
scales = [1.0]
for i in range(1, NF):
    k = ((i - 1) // KS) * KS
    r = pnp(k, i, f, scales[k])
    if r is None:
        print('pose failed at', i); NF = i; break
    R, t, ninl, n, s = r
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t          # keyframe k -> frame i
    T.append(T[k] @ np.linalg.inv(M))
    scales.append(s)
    if i % KS == 0:
        print('key %3d-%3d inliers %4d/%4d  step %.2f m  depth scale %.3f' % (k, i, ninl, n, np.linalg.norm(t), s), flush=True)

# ---- unproject and gather ----
FAR = 75.0            # far layer: building faces beyond KEEP, stored coarser in street_far.ply
FAR_VOX = 0.12
for i in range(NF): deps[i] *= scales[i]
dsc = deps
dmin = [cv2.erode(d, np.ones((21, 21), np.uint8)) for d in dsc]     # nearest surface around each pixel


def sky_mask(img):
    """Bright, flat regions connected to the top edge: the depth model gives sky a finite depth."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    flat = cv2.blur(np.abs(cv2.Laplacian(g, cv2.CV_32F)), (9, 9)) < 2.0
    cand = ((hsv[..., 2] > 170) & (hsv[..., 1] < 70) & flat).astype(np.uint8)
    n, lab = cv2.connectedComponents(cand)
    top = np.unique(lab[0]); top = top[top > 0]
    return cv2.dilate(np.isin(lab, top).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0


def gather(lo, hi, stride, drop_sky):
    us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    P, C, Dobs, F = [], [], [], []
    for i in range(NF):
        d = dsc[i]
        gy, gx = np.gradient(d)
        edge = np.hypot(gx, gy) / d                      # drop "flying" pixels on depth jumps
        dd = d[vs, us]
        ok = (dd >= lo) & (dd < hi) & (edge[vs, us] < 0.03)
        if drop_sky:
            ok &= ~sky_mask(imgs[i])[vs, us]
        u, v, dd = us[ok], vs[ok], dd[ok]
        X = np.stack([(u - W / 2) / f * dd, (v - H / 2) / f * dd, dd, np.ones_like(dd)], 1)
        P.append((X @ T[i].T)[:, :3].astype(np.float32))
        C.append(imgs[i][v, u, ::-1])
        Dobs.append(dd)
        F.append(np.full(len(dd), i, np.int16))
    return np.concatenate(P), np.concatenate(C), np.concatenate(Dobs), np.concatenate(F)


def carve(P, F, tol=0.12, need=2):
    """Free-space carving: drop points that other frames see straight through
    (ghost layers and smears left by depth / pose disagreement)."""
    votes = np.zeros(len(P), np.int8)
    for i in range(NF):
        idx = np.nonzero(F == i)[0]
        for off in (-8, -4, -2, 2, 4, 8):
            j = i + off * KS
            if j < 0 or j >= NF: continue
            Xc = (P[idx] - T[j][:3, 3].astype(np.float32)) @ T[j][:3, :3].astype(np.float32)
            z = Xc[:, 2]
            zs = np.maximum(z, 1e-3)
            u = (Xc[:, 0] / zs * f + W / 2).astype(np.int32)
            v = (Xc[:, 1] / zs * f + H / 2).astype(np.int32)
            ok = (z > 1.0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            hit = np.zeros(len(idx), bool)
            hit[ok] = z[ok] < dmin[j][v[ok], u[ok]] * (1.0 - tol)
            votes[idx] += hit
    keep = votes < need
    print('carving kept %d of %d' % (keep.sum(), len(P)), flush=True)
    return keep


def dedup(P, C, Dobs, vox):
    key = np.floor(P / vox).astype(np.int64)
    key -= key.min(0)
    k1 = (key[:, 0] << 42) | (key[:, 1] << 21) | key[:, 2]
    order = np.lexsort((Dobs, k1))                       # keep the closest observation per voxel
    k1s = k1[order]
    first = np.ones(len(order), bool); first[1:] = k1s[1:] != k1s[:-1]
    sel = order[first]
    return P[sel], C[sel]


P, C, Dobs, F = gather(0.0, KEEP, STRIDE, False)
print('raw points', len(P), flush=True)
k = carve(P, F)
P, C = dedup(P[k], C[k], Dobs[k], VOX)
print('after voxel', len(P), flush=True)
Pf, Cf, Df, Ff = gather(KEEP * 0.9, FAR, STRIDE, True)
k = carve(Pf, Ff, tol=0.2)
Pf, Cf = dedup(Pf[k], Cf[k], Df[k], FAR_VOX)
print('far layer', len(Pf), flush=True)

# ---- align: ground -> y = const, travel direction -> -z ----
cams = np.array([t[:3, 3] for t in T])
travel = cams[-1] - cams[0]
travel /= np.linalg.norm(travel)
# ground: RANSAC plane among points below the camera path (camera y is down)
cand = P[(P[:, 1] - cams[0, 1]) > 0.8]
rng = np.random.default_rng(0)
bestn, bestc = None, 0
for _ in range(400):
    a = cand[rng.integers(0, len(cand), 3)]
    n = np.cross(a[1] - a[0], a[2] - a[0])
    if np.linalg.norm(n) < 1e-6: continue
    n /= np.linalg.norm(n)
    if abs(n[1]) < 0.8: continue
    sub = cand[::50]
    c = int((np.abs((sub - a[0]) @ n) < 0.08).sum())
    if c > bestc: bestc, bestn, bestp = c, n, a[0]
up = -bestn if bestn[1] > 0 else bestn               # world up (camera y points down)
fwd = travel - up * (travel @ up); fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, up)
cam_h = float((cams[0] - bestp) @ up)
Rw = np.stack([right, up, -fwd])                      # rows: x, y, z
Q = (P - cams[0]) @ Rw.T
camsQ = (cams - cams[0]) @ Rw.T
axial = -Q[:, 2]
print('camera height %.2f m, path length %.1f m, axial range %.1f .. %.1f' % (cam_h, -camsQ[-1, 2], axial.min(), axial.max()))

# keep a clean stretch of street, centre it on z = 0
a0 = float(np.percentile(axial, 1.0)); a1 = float(np.percentile(axial, 99.0))
L = a1 - a0
ground = Q[Q[:, 1] < -cam_h + 0.4]
xc = float(np.median(ground[:, 0]))                 # street centre
shift = np.array([-xc, cam_h - EYE, a0 + L / 2])    # origin: street centre, eye height, mid-length
ok = (axial > a0) & (axial < a1) & (np.abs(Q[:, 0] - xc) < HALF_WIDTH) & (Q[:, 1] < -cam_h + 24)
Q, C = Q[ok] + shift, C[ok]
camsQ = camsQ + shift

out = os.path.join(here, 'street_scan.ply')
rec = np.empty(len(Q), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
rec['x'], rec['y'], rec['z'] = Q[:, 0], Q[:, 1], Q[:, 2]
rec['red'], rec['green'], rec['blue'] = C[:, 0], C[:, 1], C[:, 2]
with open(out, 'wb') as fh:
    fh.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n' % len(Q)).encode())
    rec.tofile(fh)
# far layer: same frame of reference, only what flanks the strip and is above street level
Qf = (Pf - cams[0]) @ Rw.T
axf = -Qf[:, 2]
okf = (axf > a0) & (axf < a1) & (np.abs(Qf[:, 0] - xc) < HALF_WIDTH * 2.2) & (Qf[:, 1] > -cam_h + 3.0) & (Qf[:, 1] < -cam_h + 110)
Qf, Cf = Qf[okf] + shift, Cf[okf]
# the detailed layer wins wherever it already has a surface (0.6 m cells, with neighbours)
def cells(X, d=(0, 0, 0)):
    k = np.floor(X / 0.6).astype(np.int64) + 4096 + np.array(d)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
occ = np.unique(np.concatenate([cells(Q[::4], (dx, dy, dz)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]))
free = ~np.isin(cells(Qf), occ)
Qf, Cf = Qf[free], Cf[free]
recf = np.empty(len(Qf), dtype=rec.dtype)
recf['x'], recf['y'], recf['z'] = Qf[:, 0], Qf[:, 1], Qf[:, 2]
recf['red'], recf['green'], recf['blue'] = Cf[:, 0], Cf[:, 1], Cf[:, 2]
with open(os.path.join(here, 'street_far.ply'), 'wb') as fh:
    fh.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n' % len(Qf)).encode())
    recf.tofile(fh)
print('wrote street_far.ply', len(Qf), 'points')
# lane the corridor camera can travel without hitting anything (knee to above head height)
body = Q[(Q[:, 1] > -0.9) & (Q[:, 1] < 3.0)]
lanes = np.arange(-HALF_WIDTH, HALF_WIDTH, 1.0)
occ = np.array([((body[:, 0] > x - 1.5) & (body[:, 0] < x + 1.5)).sum() for x in lanes])
free = lanes[occ <= max(occ.min(), 200)]
cam_x = float(free[np.argmin(np.abs(free - camsQ[:, 0].mean()))])
print('clear lane for the camera: x = %.1f' % cam_x)
meta = dict(cam_x=cam_x, far_points=int(len(Qf)), far_voxel=FAR_VOX, points=int(len(Q)), length=L, camera_height=cam_h, voxel=VOX, focal_px=f,
            x_range=[float(Q[:, 0].min()), float(Q[:, 0].max())], y_range=[float(Q[:, 1].min()), float(Q[:, 1].max())],
            cam_path=camsQ.round(2).tolist())
json.dump(meta, open(os.path.join(here, 'street_scan.json'), 'w'), indent=1)
print('wrote', out, len(Q), 'points, length %.1f m' % L)

# quick top-down and side previews
def view(a, b, name, sa, sb):
    img = np.zeros((700, 1400, 3), np.uint8)
    x = ((a - a.min()) / (a.max() - a.min()) * 1399).astype(int)
    y = (699 - (b - b.min()) / (b.max() - b.min()) * 699).astype(int)
    img[y, x] = C[:, ::-1]
    cv2.imwrite(os.path.join(here, name), img)
view(Q[:, 2], Q[:, 0], 'preview_top.jpg', 0, 0)
view(Q[:, 2], Q[:, 1], 'preview_side.jpg', 0, 0)
C = Cf
view(Qf[:, 2], Qf[:, 1], 'preview_far_side.jpg', 0, 0)
view(Qf[:, 0], Qf[:, 1], 'preview_far_front.jpg', 0, 0)
