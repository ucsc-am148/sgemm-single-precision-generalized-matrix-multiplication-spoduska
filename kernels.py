"""Student kernels for the SGEMM autograder assignment.

You implement K2 (GMEM coalescing), K3 (shared-memory blocking), K4 (1D
register tiling), and K5 (2D register tiling) inside this file. The launch
wrappers, tile-size constants, and signatures are provided — you only edit
the kernel bodies marked TODO.

K1 (naive) is given as a worked example so you have a reference for the
numba.cuda @cuda.jit signature every kernel must match.

To check correctness locally before submitting:
    python sanity_check.py

To submit: push your edits to the main branch of this assignment repo.
Each push that touches kernels.py triggers the autograder, which runs
on a Modal A100 40GB and posts your grade as a comment on the commit.
You have 5 graded submissions per assignment.
"""
import math

from numba import cuda, float32


# ── Tile constants ──────────────────────────────────────────────────
# These are tied to the launch shapes the autograder will use. Do not
# change them; the run_kN wrappers below depend on these values.

BLOCKSIZE = 32          # K1 + K2 tile

# K3 tile sizes
BM3, BN3, BK3 = 32, 32, 32

# K4 tile sizes
BM4, BN4, BK4 = 64, 64, 8
TM4 = 8

# K5 tile sizes
BM5, BN5, BK5 = 128, 128, 8
TM5, TN5 = 8, 8


# ── K1: naive (worked example, do not edit) ─────────────────────────

@cuda.jit
def sgemm_naive(A, B, C, M, N, K):
    """K1: one thread per output element. No tiling, no shared memory.
    Provided so you have a working numba.cuda kernel for reference.
    """
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x < M and y < N:
        tmp = float32(0.0)
        for i in range(K):
            tmp += A[x, i] * B[i, y]
        C[x, y] = tmp


# ── K2: GMEM coalescing (TODO) ──────────────────────────────────────

@cuda.jit
def sgemm_coalesced(A, B, C, M, N, K):
    """K2: rewrite K1 so that 32 threads in a warp end up writing to 32
    *consecutive columns* of C (and reading 32 consecutive elements of B).
    The arithmetic is identical to K1

    Launch shape (run_k2 below uses this):
        block = (BLOCKSIZE * BLOCKSIZE,)        # 1024 threads, 1D
        grid  = (ceil(M / BLOCKSIZE), ceil(N / BLOCKSIZE))

    With a 1D block of 1024 threads, threadIdx.x runs 0..1023.
    Derive (row_in_tile, col_in_tile) from threadIdx.x using integer division
    and modulo by BLOCKSIZE. 
    Be careful which one indexes the column.
    """

    col_in_tile = cuda.threadIdx.x % BLOCKSIZE   # cols vary in each warp
    row_in_tile = cuda.threadIdx.x // BLOCKSIZE  # constant in a warp

    row = cuda.blockIdx.x * BLOCKSIZE + row_in_tile
    col = cuda.blockIdx.y * BLOCKSIZE + col_in_tile

    if row < M and col < N:
        tmp = float32(0.0)
        for i in range(K):
            tmp += A[row, i] * B[i, col]
        C[row, col] = tmp

    return


# ── K3: shared-memory cache-blocking (TODO) ─────────────────────────

@cuda.jit
def sgemm_smem(A, B, C, M, N, K):
    """K3: stream the K dimension in chunks of BK3. Each block computes a
            BM3 x BN3 output tile by repeatedly:
        1. cooperatively loading a BM3 x BK3 slice of A and a BK3 x BN3
           slice of B into shared memory (one element per thread per slice),
        2. cuda.syncthreads(),
        3. dotting the row of As into the column of Bs to update one
           per-thread accumulator,
        4. cuda.syncthreads() before the next K-chunk.

    Launch shape (run_k3 below uses this):
        block = (BM3 * BN3,)                    # 1024 threads, 1D
        grid  = (ceil(M / BM3), ceil(N / BN3))

    Use cuda.shared.array((BM3, BK3), float32) for As and a similar
    (BK3, BN3) for Bs.
    Use 0.0 in the SMEM load when the global index is out of bounds.
    """

    # SMEM tiles: small caches of A and B that all threads in the block share.
    As = cuda.shared.array((BM3, BK3), float32)
    Bs = cuda.shared.array((BK3, BN3), float32)

    # Split the 1D thread index into a (row, col) inside the 32x32 tile.
    tid = cuda.threadIdx.x
    row_in_tile = tid // BN3
    col_in_tile = tid % BN3

    # Which element of C this thread will write.
    row = cuda.blockIdx.x * BM3 + row_in_tile
    col = cuda.blockIdx.y * BN3 + col_in_tile

    # Running sum for this thread's output element.
    tmp = float32(0.0)

    # Walk through K, one BK3-wide chunk at a time.
    for kt in range(0, K, BK3):

        # Each thread loads one element of A into shared memory.
        # Out-of-bounds spots get 0.0 so they don't affect the dot product.
        a_row = cuda.blockIdx.x * BM3 + row_in_tile
        a_col = kt + col_in_tile

        if a_row < M and a_col < K:
            As[row_in_tile, col_in_tile] = A[a_row, a_col]
        else:
            As[row_in_tile, col_in_tile] = float32(0.0)

        # Each thread loads one element of B into shared memory.
        b_row = kt + row_in_tile
        b_col = cuda.blockIdx.y * BN3 + col_in_tile

        if b_row < K and b_col < N:
            Bs[row_in_tile, col_in_tile] = B[b_row, b_col]
        else:
            Bs[row_in_tile, col_in_tile] = float32(0.0)

        # Wait for every thread to finish loading before anyone reads.
        cuda.syncthreads()

        # Multiply this row of As by this column of Bs and add to tmp.
        for i in range(BK3):
            tmp += As[row_in_tile, i] * Bs[i, col_in_tile]

        # Wait for everyone to finish reading before the next chunk overwrites As/Bs.
        cuda.syncthreads()

    # Write the final result to C (skip if we're past the edge of the matrix).
    if row < M and col < N:
        C[row, col] = tmp

    return


# ── K4: 1D register tiling (TODO) ───────────────────────────────────

@cuda.jit
def sgemm_1d_tile(A, B, C, M, N, K):
    """K4: extend K3 by giving each thread TM4 = 8 rows in a single column
    of the BM4 x BN4 output tile.

    Note: blockIdx.x now indexes COLUMNS of the output.
    The run_k4 wrapper below already accounts for this, but you need to compute the global (row, col)
    start of your block accordingly.

    Launch shape (run_k4 below uses this):
        block = ((BM4 * BN4) // TM4,)           # 512 threads
        grid  = (ceil(N / BN4), ceil(M / BM4))  # x = col, y = row

    Cooperative loads here are tidy: A's tile is BM4 x BK4 = 512 elements,
    B's tile is BK4 x BN4 = 512 elements, and you have 512 threads so
    exactly one element per thread per tile (so no inner-load loop)

    Use cuda.local.array(TM4, float32) for the per-thread accumulator array.
    Initialize all entries to 0.0 before the K-loop.
    """

    # SMEM tiles: As caches a 64x8 slice of A, Bs caches an 8x64 slice of B.
    As = cuda.shared.array((BM4, BK4), float32)
    Bs = cuda.shared.array((BK4, BN4), float32)

    # Axis swap: blockIdx.x picks the column tile, blockIdx.y picks the row tile.
    block_row_start = cuda.blockIdx.y * BM4
    block_col_start = cuda.blockIdx.x * BN4

    tid = cuda.threadIdx.x

    # Compute mapping: this thread owns 8 rows in a single column of the C tile.
    threadCol = tid % BN4         # which column (0..63), varies fastest in a warp
    threadRow = tid // BN4        # which 8-row strip (0..7)

    # Load mapping for A (64x8 = 512 elements, one per thread).
    innerRowA = tid // BK4
    innerColA = tid % BK4

    # Load mapping for B (8x64 = 512 elements, one per thread).
    innerRowB = tid // BN4
    innerColB = tid % BN4

    # Running sums for thread's 8 output elements (kept in registers).
    threadResults = cuda.local.array(TM4, float32)
    for m in range(TM4):
        threadResults[m] = float32(0.0)

    # Walk K in chunks of BK4=8.
    for kt in range(0, K, BK4):

        # Cooperative load of A tile (zero-fill if out of bounds).
        a_row = block_row_start + innerRowA
        a_col = kt + innerColA
        if a_row < M and a_col < K:
            As[innerRowA, innerColA] = A[a_row, a_col]
        else:
            As[innerRowA, innerColA] = float32(0.0)

        # Cooperative load of B tile (zero-fill if out of bounds).
        b_row = kt + innerRowB
        b_col = block_col_start + innerColB
        if b_row < K and b_col < N:
            Bs[innerRowB, innerColB] = B[b_row, b_col]
        else:
            Bs[innerRowB, innerColB] = float32(0.0)

        # Wait for every thread to finish loading before anyone reads SMEM.
        cuda.syncthreads()

        # Inner dot: cache one Bs value, reuse it for all 8 rows.
        # This 8x reuse on Bs is what makes K4 faster than K3.
        for k in range(BK4):
            Btmp = Bs[k, threadCol]
            for m in range(TM4):
                threadResults[m] += As[threadRow * TM4 + m, k] * Btmp

        # Wait for everyone to finish reading before the next chunk overwrites SMEM.
        cuda.syncthreads()

    # Write the 8 accumulated results to global C (skip out-of-bounds rows/cols).
    for m in range(TM4):
        out_row = block_row_start + threadRow * TM4 + m
        out_col = block_col_start + threadCol
        if out_row < M and out_col < N:
            C[out_row, out_col] = threadResults[m]

    return


# ── K5: 2D register tiling (TODO) ───────────────────────────────────

@cuda.jit
def sgemm_2d_tile(A, B, C, M, N, K):
    """K5: extend K4 to a TM5 x TN5 = 8 x 8 register tile per thread.
    Inside the inner-k loop, cache TM5 As values and TN5 Bs values into
    register arrays, then do the TM5 x TN5 outer-product update.

    Launch shape (run_k5 below uses this):
        block = ((BM5 * BN5) // (TM5 * TN5),)   # 256 threads
        grid  = (ceil(N / BN5), ceil(M / BM5))

    Cooperative loads now need a stride loop: the tile has more elements
    (BM5 * BK5 = 1024) than the block has threads (256), so each thread
    loads BM5 * BK5 / 256 = 4 elements of A per K-chunk and similarly for B.
    Pick the per-thread row stride so that consecutive threads touch
    consecutive memory addresses (= coalesced GMEM loads).

    For accumulators, use cuda.local.array((TM5, TN5), float32).
    Numba supports tuple-shaped local arrays!
    """

    # SMEM tiles: As is 128x8, Bs is 8x128.
    As = cuda.shared.array((BM5, BK5), float32)
    Bs = cuda.shared.array((BK5, BN5), float32)

    # Axis swap: blockIdx.x picks the column tile, blockIdx.y picks the row tile.
    block_row_start = cuda.blockIdx.y * BM5
    block_col_start = cuda.blockIdx.x * BN5

    tid = cuda.threadIdx.x

    # Compute mapping: each thread owns an 8x8 square of the C tile.
    # The 128x128 tile is laid out as a 16x16 grid of 8x8 squares (256 threads).
    threadRow = tid // (BN5 // TN5)   # 0..15, which 8-row strip
    threadCol = tid % (BN5 // TN5)    # 0..15, which 8-col strip

    # Load mapping for A (128x8 = 1024 elements, 4 per thread via stride loop).
    innerRowA = tid // BK5            # 0..31
    innerColA = tid % BK5             # 0..7
    strideA = ((BM5 * BN5) // (TM5 * TN5)) // BK5   # = 32 rows per pass

    # Load mapping for B (8x128 = 1024 elements, 4 per thread via stride loop).
    innerRowB = tid // BN5            # 0..1
    innerColB = tid % BN5             # 0..127
    strideB = ((BM5 * BN5) // (TM5 * TN5)) // BN5   # = 2 rows per pass

    # Per-thread 8x8 accumulator + small caches for the inner outer-product step.
    threadResults = cuda.local.array((TM5, TN5), float32)
    regM = cuda.local.array(TM5, float32)
    regN = cuda.local.array(TN5, float32)
    for m in range(TM5):
        for n in range(TN5):
            threadResults[m, n] = float32(0.0)

    # Walk K in chunks of BK5=8.
    for kt in range(0, K, BK5):

        # Cooperative load of A tile (4 passes of 256 elements each).
        for offset in range(0, BM5, strideA):
            a_row = block_row_start + innerRowA + offset
            a_col = kt + innerColA
            if a_row < M and a_col < K:
                As[innerRowA + offset, innerColA] = A[a_row, a_col]
            else:
                As[innerRowA + offset, innerColA] = float32(0.0)

        # Cooperative load of B tile (4 passes of 256 elements each).
        for offset in range(0, BK5, strideB):
            b_row = kt + innerRowB + offset
            b_col = block_col_start + innerColB
            if b_row < K and b_col < N:
                Bs[innerRowB + offset, innerColB] = B[b_row, b_col]
            else:
                Bs[innerRowB + offset, innerColB] = float32(0.0)

        # Wait for every thread to finish loading before anyone reads SMEM.
        cuda.syncthreads()

        # Inner loop: outer-product update over BK5.
        # Each inner-k step reads 8 A-values + 8 B-values from SMEM
        # and does 64 FMAs in registers — 4x better intensity than K4.
        for k in range(BK5):
            for m in range(TM5):
                regM[m] = As[threadRow * TM5 + m, k]
            for n in range(TN5):
                regN[n] = Bs[k, threadCol * TN5 + n]
            for m in range(TM5):
                for n in range(TN5):
                    threadResults[m, n] += regM[m] * regN[n]

        # Wait for everyone to finish reading before the next chunk overwrites SMEM.
        cuda.syncthreads()

    # Write the 8x8 = 64 results to global C (skip out-of-bounds rows/cols).
    for m in range(TM5):
        for n in range(TN5):
            out_row = block_row_start + threadRow * TM5 + m
            out_col = block_col_start + threadCol * TN5 + n
            if out_row < M and out_col < N:
                C[out_row, out_col] = threadResults[m, n]

    return


# ── Launch wrappers (provided — do not edit) ────────────────────────

def run_k1(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE, BLOCKSIZE)
    sgemm_naive[grid, block](A, B, C, M, N, K)


def run_k2(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE * BLOCKSIZE,)
    sgemm_coalesced[grid, block](A, B, C, M, N, K)


def run_k3(A, B, C, M, N, K):
    grid = (math.ceil(M / BM3), math.ceil(N / BN3))
    block = (BM3 * BN3,)
    sgemm_smem[grid, block](A, B, C, M, N, K)


def run_k4(A, B, C, M, N, K):
    # Axis swap: blockIdx.x indexes columns of C.
    grid = (math.ceil(N / BN4), math.ceil(M / BM4))
    block = ((BM4 * BN4) // TM4,)
    sgemm_1d_tile[grid, block](A, B, C, M, N, K)


def run_k5(A, B, C, M, N, K):
    grid = (math.ceil(N / BN5), math.ceil(M / BM5))
    block = ((BM5 * BN5) // (TM5 * TN5),)
    sgemm_2d_tile[grid, block](A, B, C, M, N, K)


# Graded kernels in the order the rubric uses (1/4 → C, 2/4 → B-, ...).
KERNELS = [
    ("k2_coalesce", run_k2),
    ("k3_smem",     run_k3),
    ("k4_1d_tile",  run_k4),
    ("k5_2d_tile",  run_k5),
]
