"""Dump teletext packets of one magazine/packet number.  usage: special.py <survey-dir> <mag> <packet> [limit]"""
import sys,os,numpy as np
from collections import Counter
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from wst2 import HAM,HERR
from analyze_helpers import txt
def load(d):
    raw=np.fromfile(os.path.join(d,'packets.bin'),np.uint8); raw=raw[:len(raw)//51*51].reshape(-1,51)
    fr=raw[:,:4].copy().view('<u4').ravel(); P=raw[:,6:]; rows=raw[:,4]
    ok=(HERR[P[:,3]]==0)&(HERR[P[:,4]]==0)
    mag=HAM[P[:,3]]&7; mag[mag==0]=8
    pkt=(HAM[P[:,3]]>>3)|(HAM[P[:,4]]<<1)
    return fr,rows,P,ok,mag,pkt
def par(b): return b&0x7f
t,m,n=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]); lim=int(sys.argv[4]) if len(sys.argv)>4 else 20
fr,rows,P,ok,mag,pkt=load(t)
sel=np.where(ok&(mag==m)&(pkt==n))[0]
print(len(sel))
for i in sel[:lim]:
    d=bytes(P[i,5:45])
    print(fr[i],rows[i],d.hex(),'|'+txt(bytes(par(x) for x in d))+'|')
