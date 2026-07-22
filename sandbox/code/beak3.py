"""Beak model for "37" — final ruleset benchmark.

Board 37 cells (radius 3), C3 ring start (2 cells per player on opposite
neighbours of the centre). B4 births (majority / 2-1-1 plurality / 2-2 -> the
absent third colour), survival 1..smax.

Beak economy: births create cells from nothing (total +1) and deaths destroy them
(total -1) — neither touches a beak. 'drop' moves a cell beak -> board, 'pick'
moves board -> beak. Beak capacity `cap`; initial beak content `init_beak`.
Since a winning chain needs 7 cells, a player starting with init_beak + 2 total
must win at least 7 - init_beak - 2 net births before a win is possible.

Usage: python3 beak3.py <cap> <ngames>
"""
import random, collections, statistics, sys
src=open('hex3life.py').read().replace('R = 4','R = 3').split("if __name__")[0]
M={}; exec(src,M)
ROT=lambda c:(c[2],c[0],c[1]); IDX=M['IDX']; N=M['N']; EMPTY=M['EMPTY']; NBRS=M['NBRS']
board0=[EMPTY]*N; cells=[(1,0,-1),(-1,0,1)]
for p in range(3):
    for c in cells: board0[IDX[c]]=p
    cells=[ROT(c) for c in cells]

def trans(board,smax,st=None):
    new=board[:]; births=[]
    for i in range(N):
        cnt=[0,0,0]
        for j in NBRS[i]:
            if board[j]!=EMPTY: cnt[board[j]]+=1
        n=sum(cnt)
        if board[i]!=EMPTY:
            if not (1<=n<=smax):
                new[i]=EMPTY
                if st is not None: st['deaths']+=1
        elif n==4:
            s=sorted(range(3),key=lambda q:-cnt[q])
            if cnt[s[0]]==2 and cnt[s[1]]==2: c=s[2]; kind='2-2'
            elif cnt[s[0]]==4: c=s[0]; kind='4-0'
            elif cnt[s[0]]==3: c=s[0]; kind='3-1'
            else: c=s[0]; kind='2-1-1'
            births.append((i,c))
            if st is not None: st['births']+=1; st[kind]+=1
    for i,c in births: new[i]=c
    return new

def legal(board,p,beak,cap):
    mv=[]
    if beak[p]>0: mv+=[('drop',i) for i in range(N) if board[i]==EMPTY]
    if beak[p]<cap: mv+=[('pick',i) for i in range(N) if board[i]==p]
    return mv

def apply_(board,p,mv,beak,smax,cap,st=None):
    b=board[:]; bk=beak[:]
    if mv[0]=='drop': b[mv[1]]=p; bk[p]-=1
    else: b[mv[1]]=EMPTY; bk[p]+=1
    return trans(b,smax,st), bk

def greedy(board,p,beak,rng,smax,cap):
    best=None; bs=None
    for mv in legal(board,p,beak,cap):
        b2,_=apply_(board,p,mv,beak,smax,cap)
        w=M['winner_after'](b2,p)
        if w==p: return mv
        if w is not None: sc=-1e8
        else:
            my=M['path_dist'](b2,p)
            opp=sorted(M['path_dist'](b2,q) for q in range(3) if q!=p)
            sc=-3*my+opp[0]+0.3*opp[1]+0.01*rng.random()
        if bs is None or sc>bs: bs,best=sc,mv
    return best

def rand(board,p,beak,rng,smax,cap):
    mv=legal(board,p,beak,cap)
    return rng.choice(mv) if mv else None

def bench(agent,n,seed0,smax,cap,init_beak):
    out=collections.Counter(); lens=[]; dens=[]
    st=collections.Counter(); ext=0; extp=0; passes=0; turns=0
    f7=[]; f7w=0; f7g=0; finals=[]; picks=0; drops=0; mx=[]
    for s in range(n):
        rng=random.Random(seed0+s)
        board=board0[:]; beak=[init_beak]*3
        seen=collections.Counter(); w=None; ff=None; ffp=None; m=init_beak+2
        for t in range(400):
            p=t%3; turns+=1
            mv=(greedy if agent=='g' else rand)(board,p,beak,rng,smax,cap)
            if mv is None:
                passes+=1
                k=(tuple(board),tuple(beak),(t+1)%3); seen[k]+=1
                if seen[k]>=3: break
                continue
            if mv[0]=='pick': picks+=1
            else: drops+=1
            board,beak=apply_(board,p,mv,beak,smax,cap,st)
            tot=[board.count(q)+beak[q] for q in range(3)]
            m=max(m,max(tot))
            if ff is None:
                for q in range(3):
                    if tot[q]>=7: ff,ffp=t+1,q; break
            dens.append(sum(1 for v in board if v!=EMPTY)/N)
            if M['winner_after'](board,p) is not None:
                w=next(q for q in range(3) if M['connected'](board,q)); out[w]+=1; break
            k=(tuple(board),tuple(beak),(t+1)%3); seen[k]+=1
            if seen[k]>=3: out['d']+=1; break
        else: out['d']+=1
        lens.append(t+1); mx.append(m)
        tot=[board.count(q)+beak[q] for q in range(3)]
        finals.append(tot)
        g=sum(1 for q in range(3) if tot[q]==0)
        if g: ext+=1; extp+=g
        if ff is not None:
            f7g+=1; f7.append(ff)
            if w==ffp: f7w+=1
    dec=sum(out[q] for q in range(3))
    sh=[100*out[q]/dec for q in range(3)] if dec else [0]*3
    chi=sum((out[q]-dec/3)**2/(dec/3) for q in range(3)) if dec else 0
    b=max(st['births'],1)
    print("  %s cap=%d: decided %d/%d (%d%%), seats %.1f/%.1f/%.1f chi2 %.1f, len %.0f, "
          "density %.2f"%(agent,cap,dec,n,round(100*dec/n),sh[0],sh[1],sh[2],chi,
          statistics.mean(lens),statistics.mean(dens)))
    print("     births %d: 4-0 %d%% | 3-1 %d%% | 2-1-1 %d%% | 2-2->third %d%%  "
          "(mixed parentage %d%%)"%(st['births'],round(100*st['4-0']/b),
          round(100*st['3-1']/b),round(100*st['2-1-1']/b),round(100*st['2-2']/b),
          round(100*(st['3-1']+st['2-1-1']+st['2-2'])/b)))
    print("     deaths %d (b/d ratio %.2f), extinction games %d%% (%.2f players), "
          "passes %d%%, picks %d%% of moves"%(st['deaths'],st['births']/max(st['deaths'],1),
          round(100*ext/n),extp/n,round(100*passes/turns),round(100*picks/max(picks+drops,1))))
    print("     reached 7 total: %d%% of games (median ply %s), that player wins %d%%; "
          "avg peak total %.1f"%(round(100*f7g/n), int(statistics.median(f7)) if f7 else '-',
          round(100*f7w/max(f7g,1)), statistics.mean(mx)))

if __name__=='__main__':
    cap=int(sys.argv[1]); n=int(sys.argv[2])
    print("BEAK cap=%d, initial beak=%d (+2 on rock = %d total; need 7 => %d net births)"
          %(cap,cap,cap+2,7-cap-2))
    bench('g',n,600000+cap,4,cap,cap)
    bench('r',max(n//2,150),700000+cap,4,cap,cap)
