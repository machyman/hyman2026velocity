"""Exact ratio R(N) with RELATIVE control errors. Absolute error on a sum growing like 3N^2 is misleading."""
import warnings
from w2_exact import moment, uni_moment
NS=[6,8,10,12,14,16,20,24,32,48,64]
print(f"{'N':>4} {'S_srt':>10} {'S_ext':>10} {'R(N)':>10} {'det':>12} {'relNC1':>9} {'relNC3':>9} {'warn':>5}")
rows=[]
for N in NS:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        nc1=abs(sum(uni_moment(N,i,1)[0] for i in range(1,N+1))-N)/N
        allp=[(i,j) for i in range(1,N+1) for j in range(i+1,N+1)]
        ex3=3.0*N*(N-1)
        nc3=abs(sum(moment(N,i,j,2,1)[0]+moment(N,i,j,1,2)[0] for i,j in allp)-ex3)/ex3
        srt=[(2*k+1,2*k+2) for k in range(N//2)]; ext=[(k+1,N-k) for k in range(N//2)]
        def AB(pr):
            A=sum(moment(N,i,j,1,1)[0] for i,j in pr)-N/2
            B=sum(moment(N,i,j,2,1)[0]+moment(N,i,j,1,2)[0] for i,j in pr)-3*N
            return A,B
        A_s,B_s=AB(srt); A_e,B_e=AB(ext); nw=len(w)
    Ss,Se=B_s/A_s,B_e/A_e; det=A_s*B_e-A_e*B_s
    rows.append((N,Ss,Se,Ss/Se,det,nc1,nc3,nw))
    print(f"{N:>4} {Ss:>10.6f} {Se:>10.6f} {Ss/Se:>10.6f} {det:>+12.2f} {nc1:>9.1e} {nc3:>9.1e} {nw:>5}")
print(f"\nworst relative control error over the sweep: {max(max(r[5],r[6]) for r in rows):.1e}")
print(f"through N=24 only:                          {max(max(r[5],r[6]) for r in rows if r[0]<=24):.1e}")
print(f"S_srt monotone increasing: {all(rows[i][1]<rows[i+1][1] for i in range(len(rows)-1))}")
print(f"S_ext monotone decreasing: {all(rows[i][2]>rows[i+1][2] for i in range(len(rows)-1))}")
print(f"gap at N=6: {rows[0][1]-rows[0][2]:.4f}; min S_srt {min(r[1] for r in rows):.4f}; max S_ext {max(r[2] for r in rows):.4f}")
