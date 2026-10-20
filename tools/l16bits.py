import sys, numpy as np
HB=17.67
def bits_of(line):
    x=line.astype(float); x=x-np.median(x[:100]); t=np.percentile(x,99)
    if t<30: return None
    x/=t
    c=np.nonzero((x[1:]>0.5)&(x[:-1]<=0.5))[0]
    if not len(c) or c[0]>400: return None
    st=c[0]+(0.5-x[c[0]])/(x[c[0]+1]-x[c[0]]) - 0.25*HB   # rising edge ~ start of first half-bit (slow rise)
    out=[]
    e=np.nonzero(x>0.5)[0][-1]
    nb=int(round((e-st)/(2*HB)))
    for k in range(nb):
        t1=st+(2*k+0.5)*HB; t2=t1+HB
        y1=np.interp(t1,np.arange(2048),x); y2=np.interp(t2,np.arange(2048),x)
        out.append('1' if y1>y2 else '0')
    return ''.join(out)
if __name__=='__main__':
    from collections import Counter
    p=sys.argv[1]; r=int(sys.argv[2])
    raw=np.memmap(p,np.uint8,'r'); a=raw[:raw.size//65536*65536].reshape(-1,32,2048)
    C=Counter(bits_of(a[f,r]) for f in range(0,len(a),7))
    for k,v in C.most_common(6): print(v,k,len(k) if k else 0)
