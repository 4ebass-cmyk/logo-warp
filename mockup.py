"""mockup.py: put a logo onto a product photo so it sits in the photo's shape and light.
Fast and low-memory: only the logo's region is processed (about 1 s for a 33 MP photo).

Job JSON:
{ "photo": "...jpg", "logo": "...png" (RGBA, alpha = print), "out": "...jpg", "preview": optional crop png,
  "center": [x,y], "width": px,
  "angle": deg of the logo rows (+ = clockwise), "angle_v": deg of the logo verticals (default = angle;
           differs on twisted or sheared surfaces, e.g. a torso: rows 8, verticals 2.5),
  "arc": px (+ = the ends drop, as on a chest or the side of a bag),
  "fold": px (max shift of the print by folds, 0 = off), "fold_blur": px (fold-map smoothness, ~20),
  "ink": "#1c1917" (one-colour print; omit to keep the logo's own colours),
  "ink_ref": "#d5d4d3" optional fabric colour the ink was measured against; otherwise the local fabric is used,
  "opacity": 1.0, "light": 1.0 (0 = flat ink, 1 = full photo light and texture through the ink), "soft": 0.8,
  "light_blur": px, how smooth the light on the ink is (auto: 1.5 for dark ink on light fabric, 30 for light ink on dark),
  "mode": "free" (default, settings above) | "quad" | "revolve",
  "knit": true   (free mode, knitted garments) read the knit lines around the logo: sets angle_v automatically,
                 lets the logo's verticals follow the knit through folds, and compresses it where the fabric turns away,
  "fold_source": "auto" (default) | "shading" | "depth"   what the fold displacement is computed from. "depth" uses a
                 small depth network (needs torch + transformers) and is not fooled by stains, shadows or existing prints;
                 "auto" picks depth on dark fabric (where shading is mostly noise) if torch is installed, else shading }

mode "quad": "quad": [[x,y] top-left, top-right, bottom-right, bottom-left] of the logo box in the photo (flat faces).
mode "revolve": bottles, mugs, tumblers, cans, cones. The shape is read off the photo:
  "revolve": {"body": [y_top, y_bottom]   photo rows where the product's outline is clean,
              "axis_rows": [[y_top, y_bottom], ...] optional rows where BOTH outline edges are visible (axis fit),
              "ring": [y_top, y_bottom]   a band holding the FRONT arc of one circular edge (rim, step, lid edge),
              "side": "both"|"left"|"right"  which outline edge gives the radius (a handle hides the other),
              "ring2": optional second circle band at another height; used to estimate the focal length when
                       the photo has no EXIF (typical for web images),
              "focal_px" or "focal_mm35": optional; otherwise EXIF, else ring2, else 50 mm equivalent}
  "center": [x,y] where the logo's middle goes, "width": its width in px measured at that point.
  The logo's rows follow the product's circles, its verticals follow the product's profile.
  The shape analysis is cached in .mockup_cache/ next to the job, so later runs on the same photo are fast.
"""
import json, sys, os, hashlib, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def to_lin(c): c=np.asarray(c,np.float32); return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def to_srgb(c): c=np.clip(c,0,1); return np.where(c<=0.0031308,c*12.92,1.055*c**(1/2.4)-0.055)
def hexrgb(h): return np.array([int(h[i:i+2],16) for i in (1,3,5)],np.float32)/255

def quad_uv(q, X, Y):
    """inverse homography: photo px -> logo (u,v) for a logo box with corners TL, TR, BR, BL"""
    A = []
    for (u, v), (x, y) in zip([(0, 0), (1, 0), (1, 1), (0, 1)], q):
        A += [[x, y, 1, 0, 0, 0, -u*x, -u*y, -u], [0, 0, 0, x, y, 1, -v*x, -v*y, -v]]
    Hm = np.linalg.svd(np.array(A, float))[2][-1].reshape(3, 3)
    w = Hm[2, 0]*X + Hm[2, 1]*Y + Hm[2, 2]
    return (Hm[0, 0]*X + Hm[0, 1]*Y + Hm[0, 2])/w, (Hm[1, 0]*X + Hm[1, 1]*Y + Hm[1, 2])/w

def focal_px(P, job):
    """focal length in full-res px and where it came from: job, EXIF, or None (to be estimated / defaulted)"""
    rv = job.get('revolve', {})
    if rv.get('focal_px'): return float(rv['focal_px']), 'job'
    if rv.get('focal_mm35'): return float(rv['focal_mm35'])/36.0*max(P.size), 'job (35 mm equiv.)'
    try:
        ex = P.getexif(); sub = ex.get_ifd(0x8769); f35 = sub.get(0xA405) or ex.get(0xA405)
    except Exception:
        f35 = None
    if f35: return float(f35)/36.0*max(P.size), 'EXIF'   # the 35 mm frame is 36 mm on its long side
    return None, None

def fit_revolve(P, job):
    """Read a body of revolution off the photo: axis, radius profile, camera tilt. Analysis runs on a 1/4 copy."""
    rv = job['revolve']; S = 4
    key = hashlib.md5(json.dumps([os.path.abspath(job['photo']), rv], sort_keys=True).encode()).hexdigest()[:12]
    cache = os.path.join(os.path.dirname(os.path.abspath(job.get('_job_path', 'x'))), '.mockup_cache', key + '.json')
    if os.path.exists(cache): return json.load(open(cache))
    sm = np.asarray(P.resize((P.width//S, P.height//S), Image.BILINEAR), np.float32)
    # background: a smooth (quadratic) colour field fitted to the frame border, so studio light falloff is not "object"
    h_, w_ = sm.shape[:2]; yy, xx = np.mgrid[0:h_, 0:w_]; b = np.zeros((h_, w_), bool); b[:10] = b[-10:] = b[:, :10] = b[:, -10:] = True
    def basis(x, y): x = x/w_ - 0.5; y = y/h_ - 0.5; return np.stack([np.ones_like(x), x, y, x*x, y*y, x*y], -1)
    coef = np.linalg.lstsq(basis(xx[b], yy[b]), sm[b], rcond=None)[0]
    bgf = basis(xx.astype(np.float32), yy.astype(np.float32)) @ coef
    obj = gaussian_filter((np.abs(sm - bgf).max(2) > 30).astype(np.float32), 1.5) > 0.5
    def edges(a, b):
        out = []
        for y in range(int(a/S), int(b/S)):
            xs = np.nonzero(obj[y])[0]
            if len(xs) > 10: out.append((y, xs.min(), xs.max()))
        return np.array(out, float)
    E = edges(*rv['body'])
    if 'axis_rows' in rv: A_ = np.concatenate([edges(a, b) for a, b in rv['axis_rows']])
    else: A_ = E
    ax = np.polyfit(A_[:, 0], (A_[:, 1] + A_[:, 2])/2, 1)          # axis x(y), image rows in 1/4 px
    rows = E[:, 0]; xc = np.polyval(ax, rows); side = rv.get('side', 'both')
    r = {'left': xc - E[:, 1], 'right': E[:, 2] - xc}.get(side, (E[:, 2] - E[:, 1])/2)
    r = gaussian_filter(r, 3)
    # rings: the front arc of a circular edge = strongest vertical brightness change inside the band
    gy = np.abs(np.gradient(gaussian_filter(sm.mean(2), 1.2), axis=0)); cyi = P.height/S/2
    def get_ring(band):
        ra, rb = int(band[0]/S), int(band[1]/S); ym = (ra + rb)/2
        xcm = np.polyval(ax, ym); rm = float(np.interp(ym, rows, r))
        xs = np.arange(int(xcm - 0.8*rm), int(xcm + 0.8*rm))
        return np.array([(x, ra + int(np.argmax(gy[ra:rb, x]*obj[ra:rb, x]))) for x in xs], float), ra, rb, rm
    def err(ring, ra, rb, rm, phi, f):  # front arc: y = y0 + k(y0)*r*cos(theta), k = sin(phi + atan((y0 - cy)/f))
        best = 1e9
        for y0r in np.arange(ra - rm*0.8, rb + 1, 0.5):
            k = np.sin(phi + np.arctan((y0r - cyi)/f)); sn = np.clip((ring[:, 0] - np.polyval(ax, y0r))/rm, -1, 1)
            best = min(best, float(np.median(np.abs(y0r + k*rm*np.sqrt(1 - sn*sn) - ring[:, 1]))))
        return best
    phis = np.radians(np.arange(-10, 45, 0.5))
    def fit_phi(R1, f):
        e = [err(*R1, p, f) for p in phis]; i = int(np.argmin(e)); return float(phis[i]), e[i]
    R1 = get_ring(rv['ring'])
    f_full, f_src = focal_px(P, job)
    if f_full is None and rv.get('ring2'):
        # no EXIF: a second circle at another height pins the focal length (how fast the curve changes with height)
        R2 = get_ring(rv['ring2']); best = None
        for mm in (18, 24, 28, 35, 42, 50, 60, 70, 85, 105, 135, 200, 300):
            fs = mm/36.0*max(P.size)/S; phi_, e1 = fit_phi(R1, fs); e2 = err(*R2, phi_, fs)   # one camera must explain both arcs
            if best is None or e1 + e2 < best[0]: best = (e1 + e2, mm)
        f_full, f_src = best[1]/36.0*max(P.size), f'estimated from 2 rings ({best[1]} mm equiv.)'
    if f_full is None:
        f_full, f_src = 50/36.0*max(P.size), 'default 50 mm equiv. (no EXIF): keep the ring near the logo height'
    f = f_full/S; phi, e = fit_phi(R1, f)
    model = dict(scale=S, ax=[float(a) for a in ax], rows=rows.tolist(), r=r.tolist(), phi=phi, f=f, cy=cyi,
                 focal_source=f_src, focal_mm35=round(f_full/max(P.size)*36, 1),
                 ring_err_px=float(e)*S, camera_tilt_deg=float(np.degrees(phi)), ring_pts=(R1[0]*S).tolist())
    os.makedirs(os.path.dirname(cache), exist_ok=True); json.dump(model, open(cache, 'w'))
    return model

def revolve_maps(model, X, Y):
    """photo px (full res) -> angle around the axis, length along the profile (px), radius (px), visible mask"""
    S = model['scale']; ax = np.array(model['ax']); rows = np.array(model['rows']); r = np.array(model['r'])
    beta = model['phi'] + np.arctan((rows - model['cy'])/model['f']); k = np.sin(beta)
    ds = np.sqrt((1/np.maximum(np.cos(beta), 0.2))**2 + np.gradient(r)**2)
    s_tab = np.concatenate([[0], np.cumsum((ds[1:] + ds[:-1])/2)])
    x = X/S; y = Y/S; y0 = y.copy()
    for _ in range(8):
        rr = np.interp(y0, rows, r); sn = np.clip((x - np.polyval(ax, y0))/np.maximum(rr, 1e-3), -1, 1)
        y0 = y - np.interp(y0, rows, k)*rr*np.sqrt(1 - sn*sn)
    rr = np.interp(y0, rows, r); sn = (x - np.polyval(ax, y0))/np.maximum(rr, 1e-3)
    return np.arcsin(np.clip(sn, -1, 1)), np.interp(y0, rows, s_tab)*S, rr*S, np.abs(sn) < 0.995

def _cache_path(job, tag, extra):
    key = hashlib.md5(json.dumps([os.path.abspath(job['photo']), tag, extra], sort_keys=True).encode()).hexdigest()[:12]
    return os.path.join(os.path.dirname(os.path.abspath(job.get('_job_path', 'x'))), '.mockup_cache', f'{tag}-{key}.npz')

def knit_field(P, job, box, step=64, win=128):
    """Knit wale direction and pitch on a coarse grid inside box (full-res px). Batched FFT; cached.
    angle (deg): + = the knit lines move left going down (same sign as angle_v); pitch: px per wale."""
    cp = _cache_path(job, 'knit', [list(map(int, box)), step, win])
    if os.path.exists(cp): return dict(np.load(cp))
    x0, y0, x1, y1 = map(int, box); g = np.asarray(P.convert('L').crop((x0, y0, x1, y1)), np.float32)
    g = g - gaussian_filter(g, 4)
    ys = np.arange(win//2, g.shape[0] - win//2, step); xs = np.arange(win//2, g.shape[1] - win//2, step)
    if len(ys) == 0 or len(xs) == 0: return None
    yy, xx = np.meshgrid(np.arange(-win//2, win//2), np.arange(-win//2, win//2), indexing='ij')
    R = np.hypot(xx, yy); band = (R > 6) & (R < win*0.45) & (yy >= 0) & ~((yy == 0) & (xx < 0))
    w2 = np.outer(np.hanning(win), np.hanning(win)).astype(np.float32)
    A = np.zeros((len(ys), len(xs))); Pt = np.zeros_like(A); Q = np.zeros_like(A)
    for i, cy_ in enumerate(ys):                          # one FFT batch per grid row keeps memory small
        patches = np.stack([g[cy_-win//2:cy_+win//2, cx_-win//2:cx_+win//2]*w2 for cx_ in xs])
        F = np.abs(np.fft.fftshift(np.fft.fft2(patches), axes=(1, 2)))*band
        for j in range(len(xs)):
            f = F[j]; k = np.unravel_index(f.argmax(), f.shape); q = f[k]/max(np.median(f[band]), 1e-6)
            fy, fx = yy[k], xx[k]
            if fx < 0: fx, fy = -fx, -fy
            ang = np.degrees(np.arctan2(fy, fx))
            if abs(ang) < 30 and q > 12: A[i, j] = ang; Pt[i, j] = win/np.hypot(fx, fy); Q[i, j] = q
    ok = Q > 0                                            # drop wrong FFT peaks: pitch far from the median, angle far off
    if ok.sum() >= 3:
        pm = np.median(Pt[ok]); am = np.median(A[ok])
        bad = ok & ((np.abs(Pt/pm - 1) > 0.2) | (np.abs(A - am) > 12)); Q[bad] = 0; A[bad] = 0; Pt[bad] = 0
    W_ = gaussian_filter(Q, 1.2) + 1e-6
    out = dict(x=(xs + x0).astype(np.float32), y=(ys + y0).astype(np.float32),
               angle=(gaussian_filter(A*Q, 1.2)/W_).astype(np.float32), pitch=(gaussian_filter(Pt*Q, 1.2)/W_).astype(np.float32),
               quality=Q.astype(np.float32))
    os.makedirs(os.path.dirname(cp), exist_ok=True); np.savez(cp, **out)
    return out

def depth_relief(P, job, box):
    """Fold relief from a small depth network on the region (518 px), plane removed, normalised. Cached."""
    cp = _cache_path(job, 'depth', list(map(int, box)))
    if os.path.exists(cp): return np.load(cp)['h']
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1'); os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
    import torch                                          # optional heavy deps: pip install torch transformers
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    name = 'depth-anything/Depth-Anything-V2-Small-hf'
    proc = AutoImageProcessor.from_pretrained(name); net = AutoModelForDepthEstimation.from_pretrained(name).eval()
    crop = P.crop(tuple(map(int, box))); sc = 518/max(crop.size)
    small = crop.resize((max(1, int(crop.width*sc)), max(1, int(crop.height*sc))), Image.BILINEAR)
    with torch.no_grad():
        d = net(**proc(images=small, return_tensors='pt')).predicted_depth[0].numpy()
    d = np.asarray(Image.fromarray(d.astype(np.float32), mode='F').resize(crop.size, Image.BILINEAR))
    h = d - gaussian_filter(d, max(crop.size)/6)          # keep folds, drop the overall slope of the body
    h = (h/max(np.percentile(np.abs(h), 98), 1e-6)).astype(np.float32)
    os.makedirs(os.path.dirname(cp), exist_ok=True); np.savez(cp, h=h)
    return h

def run(job):
    P=Image.open(job['photo']).convert('RGB'); logo=Image.open(job['logo']).convert('RGBA')
    lw,lh=logo.size; W=float(job['width']); Hh=W*lh/lw; cx,cy=job['center']; arc=float(job.get('arc',0))
    if W<=2 or lw<2 or lh<2: raise ValueError('logo too small')
    knit=None
    if job.get('knit') and job.get('mode','free')=='free':
        m_=W*0.6+Hh+150; knit=knit_field(P,job,(max(cx-m_,0),max(cy-m_,0),min(cx+m_,P.width),min(cy+m_,P.height)))
        if knit is not None and 'angle_v' not in job:
            gx_,gy_=np.meshgrid(knit['x'],knit['y']); near=(np.abs(gx_-cx)<W/2)&(np.abs(gy_-cy)<Hh)&(knit['quality']>0)
            if near.sum()>=3: job=dict(job,angle_v=float(np.median(knit['angle'][near])))
    ang=np.radians(job.get('angle',0)); angv=np.radians(job.get('angle_v',job.get('angle',0)))
    ex=np.array([np.cos(ang),np.sin(ang)]); ey=np.array([-np.sin(angv),np.cos(angv)])
    B=np.array([[ex[0],ey[0]],[ex[1],ey[1]]])
    if abs(np.linalg.det(B))<0.3: raise ValueError('angle and angle_v too far apart')
    Minv=np.linalg.inv(B).astype(np.float32)
    def fwd(u,v):
        x=(u-0.5)*W; y=(v-0.5)*Hh+arc*4*(u-0.5)**2
        return cx+x*ex[0]+y*ey[0], cy+x*ex[1]+y*ey[1]
    fold=float(job.get('fold',0)); fb=float(job.get('fold_blur',20))
    pad=int(abs(arc)+fold*2+3*fb+40)
    model=fit_revolve(P,job) if job.get('mode')=='revolve' else None
    if job.get('mode')=='quad': cs=[tuple(p) for p in job['quad']]
    elif model: R_=W*0.8+Hh; cs=[(cx-R_,cy-R_),(cx+R_,cy+R_)]
    else: cs=[fwd(u,v) for u in (0,0.5,1) for v in (0,1)]
    x0=max(int(min(p[0] for p in cs))-pad,0); x1=min(int(max(p[0] for p in cs))+pad,P.width)
    y0=max(int(min(p[1] for p in cs))-pad,0); y1=min(int(max(p[1] for p in cs))+pad,P.height)
    if x1-x0<4 or y1-y0<4: raise ValueError('logo is outside the photo')
    reg=to_lin(np.asarray(P.crop((x0,y0,x1,y1)),np.float32)/255)
    Lum=reg@np.array([0.2126,0.7152,0.0722],np.float32)
    Y,X=np.mgrid[y0:y1,x0:x1].astype(np.float32)
    # folds: the shading-derived field is a BACKWARD displacement (photo px -> where on the flat print it comes from);
    # two fixed-point steps make it consistent, a soft clamp keeps strong shadow edges from blowing letters up
    if fold:
        src_=job.get('fold_source','auto')
        if src_=='auto':
            src_='shading'
            if float(np.median(Lum))<0.06:                  # dark fabric (about sRGB 70 and below)
                try:
                    import importlib.util as _iu
                    if _iu.find_spec('torch') and _iu.find_spec('transformers'): src_='depth'
                except Exception: pass
        if src_=='depth': h=gaussian_filter(depth_relief(P,job,(x0,y0,x1,y1)),fb)   # depth steps at fold edges: smooth like shading
        else: h=gaussian_filter(Lum,fb)-gaussian_filter(Lum,fb*6)
        gy,gx=np.gradient(h)
        g=np.hypot(gx,gy); g0=max(1e-6,float(np.percentile(g,95))); sc=np.tanh(g/g0)/np.maximum(g/g0,1e-6)
        DX=fold*gx/g0*sc; DY=fold*gy/g0*sc
        # keep the logo where it was placed: remove the field's average shift under the logo
        iyc,ixc=int(np.clip(cy-y0,0,y1-y0-1)),int(np.clip(cx-x0,0,x1-x0-1)); r_=int(max(8,min(W,Hh)/2))
        DX-=float(np.median(DX[max(iyc-r_,0):iyc+r_,max(ixc-r_*2,0):ixc+r_*2])); DY-=float(np.median(DY[max(iyc-r_,0):iyc+r_,max(ixc-r_*2,0):ixc+r_*2]))
        # no tearing: where the field changes faster than the fabric can (map folding over itself), scale it down;
        # at a sharp fold the print is squeezed instead of being cut
        lim=float(job.get('fold_max_stretch',0.35))
        for _ in range(4):
            jx=np.abs(np.gradient(DX,axis=1))+np.abs(np.gradient(DX,axis=0)); jy=np.abs(np.gradient(DY,axis=1))+np.abs(np.gradient(DY,axis=0))
            k_=np.minimum(1.0,lim/np.maximum(np.maximum(jx,jy),1e-6))
            if k_.min()>0.98: break
            k_=np.minimum(k_,gaussian_filter(k_,fb/2)); DX=gaussian_filter(DX*k_,fb/4); DY=gaussian_filter(DY*k_,fb/4)
        Xs,Ys=X.copy(),Y.copy()
        from scipy.ndimage import map_coordinates as _mc
        for _ in range(2):
            cc=[np.clip(Ys-y0,0,y1-y0-1),np.clip(Xs-x0,0,x1-x0-1)]
            Xs=X+_mc(DX,cc,order=1); Ys=Y+_mc(DY,cc,order=1)
        X,Y=Xs,Ys
    mode=job.get('mode','free'); vis=None
    if mode=='quad':
        u,v=quad_uv(job['quad'],X,Y)
    elif mode=='revolve':
        th,sv,rr,vis=revolve_maps(model,X,Y)
        thc,svc,rc,_=revolve_maps(model,np.array([[cx]],np.float32),np.array([[cy]],np.float32))
        u=(th-thc[0,0])*rc[0,0]/W+0.5; v=(sv-svc[0,0])/Hh+0.5        # rows follow the circles, verticals the profile
    else:
        dx,dy=X-cx,Y-cy; xr=Minv[0,0]*dx+Minv[0,1]*dy; yr=Minv[1,0]*dx+Minv[1,1]*dy
        if knit is not None:
            # in the logo frame: fabric columns drift with the local knit angle, and the logo compresses where the
            # knit pitch shrinks (fabric turning away). Built on an 8 px grid, then interpolated per pixel.
            from scipy.ndimage import map_coordinates
            st=8.0; gxr=np.arange(-W*0.7,W*0.7+st,st); gyr=np.arange(-Hh*0.9-abs(arc),Hh*0.9+abs(arc)+st,st)
            XR,YR=np.meshgrid(gxr,gyr); IX=cx+XR*ex[0]+YR*ey[0]; IY=cy+XR*ex[1]+YR*ey[1]
            fx=np.interp(IX.ravel(),knit['x'],np.arange(len(knit['x']))).reshape(IX.shape)
            fy=np.interp(IY.ravel(),knit['y'],np.arange(len(knit['y']))).reshape(IY.shape)
            a_=map_coordinates(knit['angle'],[fy,fx],order=1,mode='nearest'); p_=map_coordinates(knit['pitch'],[fy,fx],order=1,mode='nearest')
            j0=int(np.argmin(np.abs(gxr))); i0=int(np.argmin(np.abs(gyr)))
            p0=float(np.median(p_[max(i0-2,0):i0+3,max(j0-2,0):j0+3])); ratio=np.clip(p0/np.maximum(p_,1e-3),0.6,1.4)
            drift=np.cumsum(np.tan(np.radians(a_)-angv),axis=0)*st; drift-=drift[i0:i0+1]   # column drift vs the logo's verticals
            cu=np.cumsum(ratio,axis=1)*st; cu-=cu[:,j0:j0+1]                                   # fabric length along the row
            Ufab=cu+drift*ratio
            fi=(yr-gyr[0])/st; fj=(xr-gxr[0])/st
            xr=map_coordinates(Ufab,[fi,fj],order=1,mode='nearest').astype(np.float32)
        u=xr/W+0.5; v=(yr-arc*4*(u-0.5)**2)/Hh+0.5                  # exact inverse of rotation/shear + parabola
    A=np.asarray(logo,np.float32)/255; al_src=A[...,3]
    def samp(img):
        uu=np.clip(u*(lw-1),0,lw-1.001); vv=np.clip(v*(lh-1),0,lh-1.001); i=vv.astype(np.int32); j=uu.astype(np.int32); a=uu-j; b=vv-i
        w00=(1-a)*(1-b); w01=a*(1-b); w10=(1-a)*b; w11=a*b
        if img.ndim==3: w00,w01,w10,w11=[w[...,None] for w in (w00,w01,w10,w11)]
        return img[i,j]*w00+img[i,j+1]*w01+img[i+1,j]*w10+img[i+1,j+1]*w11
    inside=((u>=0)&(u<=1)&(v>=0)&(v<=1)&(True if vis is None else vis)).astype(np.float32)
    al=samp(al_src)*inside
    if job.get('ink'):
        col=np.broadcast_to(to_lin(hexrgb(job['ink'])),reg.shape)
    else:                                                          # premultiplied sampling: no fringes from hidden RGB
        pm=samp(to_lin(A[...,:3])*al_src[...,None]); col=pm/np.maximum(samp(al_src)[...,None],1e-4)
    s=float(job.get('soft',0.8)); al=gaussian_filter(al,s) if s>0 else al
    # light: the ink takes the fabric's own light and texture (ratio to the fabric colour it sits on)
    ref=to_lin(hexrgb(job['ink_ref'])) if job.get('ink_ref') else np.array([np.median(reg[...,k][al<0.01]) for k in range(3)],np.float32)
    # the light that reaches the ink (grey, from luminance). Dark ink on light fabric soaks into it and keeps the fine
    # texture; light ink on dark fabric is an opaque layer: it only takes the broad light, not the fabric's texture/noise
    lumw=np.array([0.2126,0.7152,0.0722],np.float32); ink_l=float(np.mean(col@lumw)); fab_l=float(ref@lumw)
    sm_=float(job.get('light_blur',1.5 if ink_l<=fab_l else 30.0))
    Lg=gaussian_filter(Lum,sm_) if sm_>0 else Lum
    shade=np.clip(Lg/max(fab_l,1e-4),0.4 if ink_l<=fab_l else 0.7,1.6 if ink_l<=fab_l else 1.2)[...,None]
    k=float(job.get('light',1.0)); printed=col*(1-k+k*shade)
    a=(al*float(job.get('opacity',1.0)))[...,None]   # real prints are opaque; in linear light even 3% see-through greys black a lot
    out=to_srgb(reg*(1-a)+printed*a)
    P.paste(Image.fromarray((out*255+0.5).astype(np.uint8)),(x0,y0))
    P.save(job['out'],quality=93)
    if job.get('preview'): P.crop((max(x0-150,0),max(y0-150,0),min(x1+150,P.width),min(y1+150,P.height))).save(job['preview'])
    return (x0,y0,x1,y1)

if __name__=='__main__':
    job=json.load(open(sys.argv[1],encoding='utf8')); job['_job_path']=sys.argv[1]; print(run(job))
