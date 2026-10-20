import numpy as np
FS=35.46895e6; HB=FS/5e6
CRI=[(0xAAAA8A99>>(31-i))&1 for i in range(32)]
n=np.arange(-24,25); LP=np.sinc(2*3.0e6/FS*n)*np.hamming(len(n)); LP/=LP.sum()
def _tm():
    m=int(32*HB)+1; t=np.arange(m)/HB
    w=np.array(CRI,float)[np.clip(t.astype(int),0,31)]-.5
    k=np.arange(-8,9); ker=np.exp(-.5*(k/(1.0*HB))**2); w=np.convolve(w,ker/ker.sum(),'same')
    return w-w.mean()
TM=_tm()
def decode(line):
    x=np.convolve(line.astype(float),LP,'same')
    black=np.median(x[:100]); top=np.percentile(x,99)
    if top-black<25: return None
    x=(x-black)/(top-black)-0.5
    seg=x[50:700]
    c=np.correlate(seg,TM,'valid')/np.sqrt((TM**2).sum())
    st=50+int(np.argmax(c))
    best=None
    for sh in np.linspace(-0.3,0.3,4):
        s0=st+(32+sh)*HB
        t1=s0+(np.arange(104)*2+0.5)*HB; t2=t1+HB
        if t2[-1]>2040: return None
        y1=np.interp(t1,np.arange(2048),x); y2=np.interp(t2,np.arange(2048),x)
        d=y1-y2; conf=np.abs(d).min()
        if best is None or conf>best[0]: best=(conf,d)
    bits=(best[1]>0).astype(np.uint8)
    return np.packbits(bits), best[0], c.max()
def parse(b):
    # b = bytes 3..15 of the VPS line (13 bytes)
    B=lambda k:int(b[k-3])
    cni=((B(13)&3)<<10)|((B(14)&0xc0)<<2)|(B(11)&0xc0)|(B(14)&0x3f)
    pil=((B(11)&0x3f)<<14)|(B(12)<<6)|(B(13)>>2)
    day=(pil>>15)&31; mon=(pil>>11)&15; hr=(pil>>6)&31; mi=pil&63
    return dict(cni=cni,day=day,mon=mon,hour=hr,min=mi,pcs=B(5)>>6,pty=B(15),b5=B(5))
