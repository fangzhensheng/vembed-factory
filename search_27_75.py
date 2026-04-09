import math
for N in range(1, 4096):
    for dims in range(1, 10):
        loss = math.log(N) * dims
        if abs(loss - 27.7500) < 0.05:
            print(f"N={N}, dims={dims}, log(N)*dims={loss}")
