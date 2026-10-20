"""Majority-vote text of early-format packet 8/30 per tape.  usage: p830.py <survey-root>"""
import sys,os,glob,numpy as np
from collections import Counter
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from wst2 import HAM,HERR
from analyze_helpers import txt
for d in sorted(glob.glob(sys.argv[1]+'/*/')):
    raw=np.fromfile(d+'packets.bin',np.uint8); raw=raw[:len(raw)//51*51].reshape(-1,51)
    P=raw[:,6:]; ok=(HERR[P[:,3]]==0)&(HERR[P[:,4]]==0)
    mag=HAM[P[:,3]]&7; mag[mag==0]=8; pkt=(HAM[P[:,3]]>>3)|(HAM[P[:,4]]<<1)
    s=np.where(ok&(mag==8)&(pkt==30))[0]
    if len(s)<5: continue
    D=P[s,5:45]&0x7f
    # bytewise majority
    maj=bytes(Counter(D[:,i]).most_common(1)[0][0] for i in range(40))
    fr=raw[s,:4].copy().view('<u4').ravel()
    print(os.path.basename(d[:-1]).replace('_fs200',''),len(s),'|'+txt(maj)+'|')
    print('   first:','|'+txt(bytes(D[0]))+'|',' last:','|'+txt(bytes(D[-1]))+'|')
