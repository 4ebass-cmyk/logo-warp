"""Fit the product face plane from a known rectangle in the photo, check it, and output the logo quad for warp.js.

How it works:
1. Find 4 corners of something on the face whose REAL proportions are known (screen 4:3, a label, a button grid).
2. Homography H: face units -> photo px. Everything on that plane now has a correct perspective position.
3. Validate on other features that must be level/parallel in reality (button rows): their v difference in face
   units should be ~0. If not, the corners are wrong; fix them before placing anything.
4. Logo size in face units = real mm * (reference units / reference mm). Place, project 4 corners, feed warp.js quad.
5. Save blocking.png (projected face grid + logo box) and look at it before baking.

Usage: python fitplane.py photo.png  (edit REF / LOGO below)
"""
import sys, json
import numpy as np
from PIL import Image, ImageDraw

PHOTO = sys.argv[1] if len(sys.argv) > 1 else '../viz/console-photo.png'
REF = {'corners': [(51, 124), (390, 67), (389, 433), (51, 440)],   # TL, TR, BR, BL of the reference rectangle (px)
       'units': (4.0, 3.0), 'width_mm': 71.1}                     # its real proportion and real width
LOGO = {'width_mm': 40.0, 'aspect': 30 / 153, 'center_px': (225, 856)}   # where it goes, roughly (px)
CHECKS = [((295.2, 596.7), (375.1, 600.6)), ((179.2, 700.7), (245.8, 707.8))]   # pairs that are level in reality

def homog(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
    _, _, Vt = np.linalg.svd(np.array(A, float)); return Vt[-1].reshape(3, 3) / Vt[-1][-1]

SW, SH = REF['units']
H = homog([(0, 0), (SW, 0), (SW, SH), (0, SH)], REF['corners']); Hi = np.linalg.inv(H)
fwd = lambda u, v: (lambda p: p[:2] / p[2])(H @ [u, v, 1])
inv = lambda x, y: (lambda p: p[:2] / p[2])(Hi @ [x, y, 1])
for a, b in CHECKS:
    print('level check (face units, ~0 is good):', round(float(inv(*b)[1] - inv(*a)[1]), 3))
lw = SW * LOGO['width_mm'] / REF['width_mm']; lh = lw * LOGO['aspect']
uc, vc = inv(*LOGO['center_px']); uc = SW / 2          # centred on the face axis
quad = [fwd(uc - lw / 2, vc - lh / 2), fwd(uc + lw / 2, vc - lh / 2), fwd(uc + lw / 2, vc + lh / 2), fwd(uc - lw / 2, vc + lh / 2)]
json.dump([[float(x), float(y)] for x, y in quad], open('quad.json', 'w'))
img = Image.open(PHOTO).convert('RGBA'); ov = Image.new('RGBA', img.size); d = ImageDraw.Draw(ov)
for u in np.arange(-0.4, SW + 0.41, 0.2): d.line([tuple(fwd(u, -0.3)), tuple(fwd(u, SH * 2.6))], fill=(0, 255, 0, 110))
for v in np.arange(-0.3, SH * 2.6, 0.2): d.line([tuple(fwd(-0.4, v)), tuple(fwd(SW + 0.4, v))], fill=(0, 255, 0, 110))
d.polygon([tuple(q) for q in quad], outline=(255, 255, 0, 255))
Image.alpha_composite(img, ov).save('blocking.png'); print('quad.json + blocking.png written')
