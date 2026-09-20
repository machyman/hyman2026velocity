"""Route D swept over even N: the determinant, exactly, with NC1 as the control."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from w2_exact import moment, uni_moment
print(f"{'N':>4} {'NC1 err':>10} {'A_srt':>12} {'A_ext':>12} {'S_srt':>11} {'S_ext':>11} {'determinant':>14} {'R(N)':>9}")
for N in [6, 8, 10, 12, 14, 16, 20, 24]:
    nc1 = abs(sum(uni_moment(N,i,1)[0] for i in range(1,N+1)) - N)
    srt = [(2*k+1, 2*k+2) for k in range(N//2)]
    ext = [(k+1, N-k)     for k in range(N//2)]
    def AB(pairs):
        A = sum(moment(N,i,j,1,1)[0] for i,j in pairs) - N/2
        B = sum(moment(N,i,j,2,1)[0] + moment(N,i,j,1,2)[0] for i,j in pairs) - 3*N
        return A, B
    A_s, B_s = AB(srt); A_e, B_e = AB(ext)
    det = A_s*B_e - A_e*B_s
    print(f"{N:>4} {nc1:>10.1e} {A_s:>12.6f} {A_e:>12.6f} {B_s/A_s:>11.6f} {B_e/A_e:>11.6f} "
          f"{det:>+14.6f} {(B_s/A_s)/(B_e/A_e):>9.6f}")
