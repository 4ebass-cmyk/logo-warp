"""surfmesh: generic surface map for warping artwork onto any object in a photo.

The surface (fabric, label, panel, bottle wrap...) gets coordinates (u,v). A grid of nodes holds the image
position of every (u,v) node; the map is fitted by linear least squares to whatever is known:
  iso(points, 'u'|'v', value)   points that lie on a line of constant u or v (print rows, seams, hems, edges)
  pin(uv, xy)                   exact correspondences (artwork point -> photo point)
  flow(points, dirs, 'u'|'v')   local image direction of an iso-line (knit wales, corduroy ribs, stitching)
plus a bending penalty that keeps the map smooth where nothing is known. Along-line parameters of iso
points are unknown; they are re-estimated by projection each iteration (alternating optimisation).
Init with any callable uv->xy (homography, cylinder, previous fit)."""
import numpy as np

class SurfMesh:
    def __init__(self, u0, u1, v0, v1, nu=25, nv=25):
        self.U=np.linspace(u0,u1,nu); self.V=np.linspace(v0,v1,nv); self.nu,self.nv=nu,nv
        self.X=np.zeros((nv,nu)); self.Y=np.zeros((nv,nu)); self.iso=[]; self.pins=[]; self.flows=[]
    # ---------- setup
    def init(self, f):
        UU,VV=np.meshgrid(self.U,self.V); self.X,self.Y=f(UU,VV); self.X0,self.Y0=self.X.copy(),self.Y.copy()   # prior shape
    def add_iso(self, pts, axis, value, w=1.0): self.iso.append((np.asarray(pts,float),axis,float(value),w))
    def add_pin(self, uv, xy, w=4.0): self.pins.append((np.asarray(uv,float),np.asarray(xy,float),w))
    def add_flow(self, pts, dirs, axis, w=1.0): self.flows.append((np.asarray(pts,float),np.asarray(dirs,float),axis,w))
    # ---------- evaluation
    def _cell(self, u, v):
        fu=np.clip((np.asarray(u)-self.U[0])/(self.U[1]-self.U[0]),0,self.nu-1.0001); fv=np.clip((np.asarray(v)-self.V[0])/(self.V[1]-self.V[0]),0,self.nv-1.0001)
        i=fv.astype(int); j=fu.astype(int); return i,j,fu-j,fv-i
    def fwd(self, u, v):
        i,j,a,b=self._cell(u,v); X,Y=self.X,self.Y
        bl=lambda M: M[i,j]*(1-a)*(1-b)+M[i,j+1]*a*(1-b)+M[i+1,j]*(1-a)*b+M[i+1,j+1]*a*b
        return bl(X),bl(Y)
    def project(self, pts, axis, value, n=400):
        """free parameter of each point on the iso-line (nearest point of the current curve)"""
        s=np.linspace(self.U[0] if axis=='v' else self.V[0], self.U[-1] if axis=='v' else self.V[-1], n)
        cx,cy=self.fwd(s,np.full(n,value)) if axis=='v' else self.fwd(np.full(n,value),s)
        d=(pts[:,0,None]-cx[None])**2+(pts[:,1,None]-cy[None])**2; k=d.argmin(1)
        # refine between neighbours
        k0=np.clip(k-1,0,n-1); k1=np.clip(k+1,0,n-1); return s[k]+0*(s[k1]-s[k0])
    # ---------- solve
    def solve(self, iters=12, bend=1.0, prior=True, verbose=False):
        """prior=True: keep the bending of the init shape (rigid: cylinder, cone); False: bend toward flat (soft goods)"""
        nu,nv=self.nu,self.nv; N=nu*nv; idx=lambda i,j: i*nu+j
        for it in range(iters):
            rows=[]; rhs=[]                        # each row acts on one coordinate (shared matrix for X and Y)
            data=[]
            for pts,axis,val,w in self.iso:
                s=self.project(pts,axis,val)
                uv=np.c_[s,np.full(len(s),val)] if axis=='v' else np.c_[np.full(len(s),val),s]
                data.append((uv,pts,w))
            for uv,xy,w in self.pins: data.append((uv[None] if uv.ndim==1 else uv, xy[None] if xy.ndim==1 else xy, w))
            A=[];bx=[];by=[]
            for uv,xy,w in data:
                i,j,a,b=self._cell(uv[:,0],uv[:,1])
                for k in range(len(uv)):
                    r=np.zeros(N); r[idx(i[k],j[k])]=(1-a[k])*(1-b[k]); r[idx(i[k],j[k]+1)]=a[k]*(1-b[k]); r[idx(i[k]+1,j[k])]=(1-a[k])*b[k]; r[idx(i[k]+1,j[k]+1)]=a[k]*b[k]
                    A.append(r*w); bx.append(xy[k,0]*w); by.append(xy[k,1]*w)
            # bending: second differences in u, v and the mixed term
            sc=bend*np.sqrt(len(A)/max(N,1))
            for i in range(nv):
                for j in range(nu):
                    rr=[]
                    if 0<j<nu-1: r=np.zeros(N); r[idx(i,j-1)]=1; r[idx(i,j)]=-2; r[idx(i,j+1)]=1; rr.append((r,sc))
                    if 0<i<nv-1: r=np.zeros(N); r[idx(i-1,j)]=1; r[idx(i,j)]=-2; r[idx(i+1,j)]=1; rr.append((r,sc))
                    if i<nv-1 and j<nu-1: r=np.zeros(N); r[idx(i,j)]=1; r[idx(i,j+1)]=-1; r[idx(i+1,j)]=-1; r[idx(i+1,j+1)]=1; rr.append((r,sc*1.4))
                    for r,k in rr:      # bending relative to the prior shape (plane, cylinder, cone...), not to zero
                        A.append(r*k); bx.append(k*r@self.X0.ravel() if prior else 0); by.append(k*r@self.Y0.ravel() if prior else 0)
            A=np.array(A); X=np.linalg.lstsq(A,np.array(bx),rcond=None)[0]; Y=np.linalg.lstsq(A,np.array(by),rcond=None)[0]
            self.X=X.reshape(nv,nu); self.Y=Y.reshape(nv,nu)
            if verbose: print('iter',it,'data rms px',round(self.residual()[0],2))
        return self
    def residual(self):
        r=[]
        for pts,axis,val,w in self.iso:
            s=self.project(pts,axis,val); x,y=(self.fwd(s,np.full(len(s),val)) if axis=='v' else self.fwd(np.full(len(s),val),s))
            r+=list(np.hypot(x-pts[:,0],y-pts[:,1]))
        r=np.array(r); return np.sqrt((r**2).mean()), r
    # ---------- inverse map (image -> uv) by rasterising the mesh triangles
    def inverse(self, x0, y0, x1, y1, sub=4):
        W,H=x1-x0,y1-y0; Uo=np.full((H,W),np.nan); Vo=np.full((H,W),np.nan)
        us=np.linspace(self.U[0],self.U[-1],(self.nu-1)*sub+1); vs=np.linspace(self.V[0],self.V[-1],(self.nv-1)*sub+1)
        UU,VV=np.meshgrid(us,vs); XX,YY=self.fwd(UU,VV)
        for i in range(len(vs)-1):
            for j in range(len(us)-1):
                for tri in (((i,j),(i,j+1),(i+1,j)),((i+1,j+1),(i+1,j),(i,j+1))):
                    P=np.array([[XX[a],YY[a]] for a in tri]); Q=np.array([[UU[a],VV[a]] for a in tri])
                    mn=np.floor(P.min(0)).astype(int); mx=np.ceil(P.max(0)).astype(int)
                    mn=np.maximum(mn,[x0,y0]); mx=np.minimum(mx,[x1-1,y1-1])
                    if (mx<mn).any(): continue
                    gx,gy=np.meshgrid(np.arange(mn[0],mx[0]+1),np.arange(mn[1],mx[1]+1))
                    T=np.array([P[1]-P[0],P[2]-P[0]]).T; det=np.linalg.det(T)
                    if abs(det)<1e-9: continue
                    l=np.linalg.solve(T,np.stack([gx.ravel()-P[0,0],gy.ravel()-P[0,1]]))
                    ok=(l[0]>=-1e-6)&(l[1]>=-1e-6)&(l[0]+l[1]<=1+1e-6)
                    uu=Q[0,0]+l[0]*(Q[1,0]-Q[0,0])+l[1]*(Q[2,0]-Q[0,0]); vv=Q[0,1]+l[0]*(Q[1,1]-Q[0,1])+l[1]*(Q[2,1]-Q[0,1])
                    yy=gy.ravel()[ok]-y0; xx=gx.ravel()[ok]-x0; Uo[yy,xx]=uu[ok]; Vo[yy,xx]=vv[ok]
        return Uo,Vo

def homography(src, dst):
    A=[]
    for (x,y),(u,v) in zip(src,dst): A+=[[x,y,1,0,0,0,-u*x,-u*y,-u],[0,0,0,x,y,1,-v*x,-v*y,-v]]
    _,_,Vt=np.linalg.svd(np.array(A,float)); H=Vt[-1].reshape(3,3); return H/H[2,2]
def hmap(H):
    def f(u,v):
        p=H@np.stack([np.ravel(u),np.ravel(v),np.ones(np.size(u))]); return (p[0]/p[2]).reshape(np.shape(u)),(p[1]/p[2]).reshape(np.shape(u))
    return f
