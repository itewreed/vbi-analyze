"""Decode all 15 data words of the line-16 source data line and line-329 section data line.

usage: dz.py <survey-root>   (directory containing one survey output folder per tape)"""
import sys,glob,os
root=sys.argv[1]
from collections import Counter,defaultdict
rev=lambda b:int(f'{b:08b}'[::-1],2)
def text(lines,w,per=8):
    best=None
    for per in (8,16,10,12,20,24,32):
        pos=defaultdict(Counter)
        for fr,b in lines: pos[fr%per][rev(b[w-3])&0x7f]+=1
        s=''.join(chr(pos[i].most_common(1)[0][0]) if pos[i] else '?' for i in range(per))
        q=sum(pos[i].most_common(1)[0][1] for i in range(per) if pos[i])/len(lines)
        if best is None or q>best[0]+0.05: best=(q,s)
    return best
for d in sorted(glob.glob(root+'/*/')):
    f=d+'vps.txt'
    if not os.path.exists(f): continue
    L=defaultdict(list)
    for l in open(f):
        fr,r,h,c=l.split(); L[int(r)].append((int(fr),bytes.fromhex(h)))
    name=os.path.basename(d[:-1]).replace('_fs200','')
    for r in sorted(L):
        ls=L[r]
        if len(ls)<200: continue
        out=[f'{name} Z{"16" if r==9 else "329"} n={len(ls)}']
        for w in (3,5,6,8,9,10,15):
            C=Counter(rev(b[w-3]) for _,b in ls).most_common(3)
            out.append(f'W{w}:'+','.join(f'{k:02x}({v*100//len(ls)}%)' for k,v in C))
        for w in (4,7):
            q,s=text(ls,w); out.append(f'W{w}txt[{q*100:.0f}%]|{s}|')
        print(' '.join(out))
