import numpy as np
FS=35.46895e6; SPB=FS/6.9375e6
def ham_table():
    t=np.full(256,-1,int); e=np.full(256,9,int)
    for d in range(16):
        d1,d2,d3,d4=[(d>>i)&1 for i in range(4)]
        p1=1^d1^d3^d4; p2=1^d1^d2^d4; p3=1^d1^d2^d3
        bits=[p1,d1,p2,d2,p3,d3,p4:=0,d4]
        b=p1|d1<<1|p2<<2|d2<<3|p3<<4|d3<<5|d4<<7
        p4=1^(bin(b).count('1')&1); b|=p4<<6
        t[b]=d; e[b]=0
        for i in range(8):
            if e[b^(1<<i)]>1: t[b^(1<<i)]=d; e[b^(1<<i)]=1
    return t,e
HAM,HERR=ham_table()
n=np.arange(-24,25); LP=np.sinc(2*4.5e6/FS*n)*np.hamming(len(n)); LP/=LP.sum()
def tmpl():
    bits=np.array([(b>>i)&1 for b in (0x55,0x55,0x27) for i in range(8)],float)-.5
    m=int(24*SPB)+1; t=np.arange(m)/SPB
    w=bits[np.clip(t.astype(int),0,23)]
    k=np.arange(-8,9); ker=np.exp(-.5*(k/(0.8*SPB))**2); w=np.convolve(w,ker/ker.sum(),'same')
    return w-w.mean()
TM=tmpl()
def slice_line(line, thr_mode='mid'):
    x=np.convolve(line.astype(float),LP,'same')
    black=np.median(x[:100]); top=np.percentile(x,99)
    if top-black<25: return None
    x=(x-black)/(top-black)
    sm=np.convolve(x,np.ones(9)/9,'same'); ab=np.nonzero(sm[40:700]>0.3)[0]
    if not len(ab): return None
    on=40+ab[0]; lo=max(0,on-40)
    seg=x[lo:on+40+len(TM)]
    c=np.correlate(seg-seg.mean(),TM,'valid')
    st=lo+int(np.argmax(c))
    best=None
    for sh in np.linspace(-0.4,0.4,5):
        t=st+(np.arange(360)+0.5+sh)*SPB
        if t[-1]>=2047: return None
        y=np.interp(t,np.arange(2048),x)
        bits=(y>0.45).astype(np.uint8)
        b=np.packbits(bits.reshape(-1,8)[:,::-1],axis=1).ravel()
        score=(b[0]==0x55)+(b[1]==0x55)+2*(b[2]==0x27)-HERR[b[3]]-HERR[b[4]]
        if best is None or score>best[0]: best=(score,b)
    return best[1]
def mrag(b):
    a=HAM[b[3]]; c=HAM[b[4]]
    if a<0 or c<0: return None
    v=a|c<<4; mag=v&7; pk=v>>3
    return (mag if mag else 8), pk
