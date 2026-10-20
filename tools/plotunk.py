import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=sys.argv[1]; rows=[int(x) for x in sys.argv[2].split(',')]; out=sys.argv[3]
x=np.load(d+'/extra.npz')
fig,ax=plt.subplots(len(rows),1,figsize=(18,2.4*len(rows))); ax=np.atleast_1d(ax)
for i,r in enumerate(rows):
    u=x[f'unk{r}']
    for k in np.linspace(0,len(u)-1,min(3,len(u))).astype(int): ax[i].plot(u[k],lw=.6)
    if f'tmpl{r}' in x: ax[i].plot(x[f'tmpl{r}'],'k',lw=.4,alpha=.5)
    ax[i].set_title(f'row {r} unknown n={len(u)}',fontsize=8); ax[i].set_xlim(0,2048)
plt.tight_layout(); plt.savefig(out,dpi=65)
