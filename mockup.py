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
  "opacity": 1.0, "light": 1.0 (0 = flat ink, 1 = full photo light and texture through the ink), "soft": 0.8 }
"""
import json, sys, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def to_lin(c): c=np.asarray(c,np.float32); return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def to_srgb(c): c=np.clip(c,0,1); return np.where(c<=0.0031308,c*12.92,1.055*c**(1/2.4)-0.055)
def hexrgb(h): return np.array([int(h[i:i+2],16) for i in (1,3,5)],np.float32)/255

def run(job):
    P=Image.open(job['photo']).convert('RGB'); logo=Image.open(job['logo']).convert('RGBA')
    lw,lh=logo.size; W=float(job['width']); Hh=W*lh/lw; cx,cy=job['center']; arc=float(job.get('arc',0))
    if W<=2 or lw<2 or lh<2: raise ValueError('logo too small')
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
    cs=[fwd(u,v) for u in (0,0.5,1) for v in (0,1)]
    x0=max(int(min(p[0] for p in cs))-pad,0); x1=min(int(max(p[0] for p in cs))+pad,P.width)
    y0=max(int(min(p[1] for p in cs))-pad,0); y1=min(int(max(p[1] for p in cs))+pad,P.height)
    if x1-x0<4 or y1-y0<4: raise ValueError('logo is outside the photo')
    reg=to_lin(np.asarray(P.crop((x0,y0,x1,y1)),np.float32)/255)
    Lum=reg@np.array([0.2126,0.7152,0.0722],np.float32)
    Y,X=np.mgrid[y0:y1,x0:x1].astype(np.float32)
    # folds: the shading-derived field is a BACKWARD displacement (photo px -> where on the flat print it comes from);
    # two fixed-point steps make it consistent, a soft clamp keeps strong shadow edges from blowing letters up
    if fold:
        h=gaussian_filter(Lum,fb)-gaussian_filter(Lum,fb*6); gy,gx=np.gradient(h)
        g=np.hypot(gx,gy); g0=max(1e-6,float(np.percentile(g,95))); sc=np.tanh(g/g0)/np.maximum(g/g0,1e-6)
        DX=fold*gx/g0*sc; DY=fold*gy/g0*sc
        Xs,Ys=X.copy(),Y.copy()
        for _ in range(2):
            ix=np.clip((Xs-x0).astype(int),0,x1-x0-1); iy=np.clip((Ys-y0).astype(int),0,y1-y0-1)
            Xs=X+DX[iy,ix]; Ys=Y+DY[iy,ix]
        X,Y=Xs,Ys
    dx,dy=X-cx,Y-cy; xr=Minv[0,0]*dx+Minv[0,1]*dy; yr=Minv[1,0]*dx+Minv[1,1]*dy
    u=xr/W+0.5; v=(yr-arc*4*(u-0.5)**2)/Hh+0.5                      # exact inverse of rotation/shear + parabola
    A=np.asarray(logo,np.float32)/255; al_src=A[...,3]
    def samp(img):
        uu=np.clip(u*(lw-1),0,lw-1.001); vv=np.clip(v*(lh-1),0,lh-1.001); i=vv.astype(np.int32); j=uu.astype(np.int32); a=uu-j; b=vv-i
        w00=(1-a)*(1-b); w01=a*(1-b); w10=(1-a)*b; w11=a*b
        if img.ndim==3: w00,w01,w10,w11=[w[...,None] for w in (w00,w01,w10,w11)]
        return img[i,j]*w00+img[i,j+1]*w01+img[i+1,j]*w10+img[i+1,j+1]*w11
    inside=((u>=0)&(u<=1)&(v>=0)&(v<=1)).astype(np.float32)
    al=samp(al_src)*inside
    if job.get('ink'):
        col=np.broadcast_to(to_lin(hexrgb(job['ink'])),reg.shape)
    else:                                                          # premultiplied sampling: no fringes from hidden RGB
        pm=samp(to_lin(A[...,:3])*al_src[...,None]); col=pm/np.maximum(samp(al_src)[...,None],1e-4)
    s=float(job.get('soft',0.8)); al=gaussian_filter(al,s) if s>0 else al
    # light: the ink takes the fabric's own light and texture (ratio to the fabric colour it sits on)
    ref=to_lin(hexrgb(job['ink_ref'])) if job.get('ink_ref') else np.array([np.median(reg[...,k][al<0.01]) for k in range(3)],np.float32)
    shade=np.clip(reg/np.maximum(ref,1e-4),0,2.0)
    k=float(job.get('light',1.0)); printed=col*(1-k+k*shade)
    a=(al*float(job.get('opacity',1.0)))[...,None]   # real prints are opaque; in linear light even 3% see-through greys black a lot
    out=to_srgb(reg*(1-a)+printed*a)
    P.paste(Image.fromarray((out*255+0.5).astype(np.uint8)),(x0,y0))
    P.save(job['out'],quality=93)
    if job.get('preview'): P.crop((max(x0-150,0),max(y0-150,0),min(x1+150,P.width),min(y1+150,P.height))).save(job['preview'])
    return (x0,y0,x1,y1)

if __name__=='__main__':
    print(run(json.load(open(sys.argv[1],encoding='utf8'))))
