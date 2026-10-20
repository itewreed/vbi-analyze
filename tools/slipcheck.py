"""Re-frames the packets of one magazine/packet number by -2..+2 bits and prints the framing with the best odd parity.
Shows whether an odd packet type is really a bit-slipped normal packet.  usage: slipcheck.py <survey-dir> <mag> <packet> [limit]"""
import sys,numpy as np
sys.path.insert(0,'/home/user/vbi-analyze/tools')
from wst2 import HAM,HERR
d=sys.argv[1]; want=int(sys.argv[2]); pnum=int(sys.argv[3]); lim=int(sys.argv[4]) if len(sys.argv)>4 else 40
raw=np.fromfile(d+'/packets.bin',np.uint8); raw=raw[:len(raw)//51*51].reshape(-1,51)
fr=raw[:,:4].copy().view('<u4').ravel(); rows=raw[:,4]; P=raw[:,6:]
def bits(b): return [(x>>i)&1 for x in b for i in range(8)]
def byts(bs): return bytes(sum(bs[j+i]<<i for i in range(8)) for j in range(0,len(bs)-7,8))
def txt(b): return ''.join(chr(c&0x7f) if 32<=(c&0x7f)<127 else '.' for c in b)
def parity_ok(b): return sum(bin(c).count('1')&1 for c in b)/len(b)
n=0
for i in range(len(P)):
    p=bytes(P[i])
    if HERR[p[3]] or HERR[p[4]]: continue
    m=HAM[p[3]]&7 or 8; pk=(HAM[p[3]]>>3)|(HAM[p[4]]<<1)
    if not (m==want and pk==pnum): continue
    bs=bits(p)
    out=[]
    for k in (-2,-1,0,1,2):
        s=bs[k:] if k>0 else [0]*(-k)+bs
        b=byts(s)
        out.append((k,parity_ok(b[5:45]),b))
    best=max(out,key=lambda x:x[1])
    k,po,b=best
    mag=(HAM[b[3]]&7 or 8) if not HERR[b[3]] else '?'
    row=((HAM[b[3]]>>3)|(HAM[b[4]]<<1)) if not (HERR[b[3]] or HERR[b[4]]) else '?'
    print(fr[i],rows[i],'shift',k,'oddpar %.2f'%po,'orig %.2f'%out[2][1],'-> M%s/%s'%(mag,row),'|'+txt(b[5:45])+'|')
    n+=1
    if n>=lim: break
