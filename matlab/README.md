# MATLAB GRAND (reference implementation) in this project

The supervisor asked that the GRAND decoder used for the thesis numbers be Ken Duffy's
MATLAB code, not a re-implementation. This folder makes that possible without copying
his code into the repository (it is distributed under the "GRAND Codebase Non-Commercial
Academic Research Use License"; cite Duffy ICASSP 2021 and IEEE TSP 2022).

## 1. Get the reference code (once)

```
bash matlab/fetch_grand_matlab.sh        # clones into matlab/GRAND-MATLAB (gitignored)
```

or clone https://github.com/kenrduffy/GRAND-MATLAB manually into `matlab/GRAND-MATLAB`.
The files needed are `bin_ORBGRAND1.m`, `bin_ORBGRAND.m` and `landslide.m`.

## 2. Verify that the Python port is the same algorithm (bit-exact)

ORBGRAND is deterministic, so the port and the reference must give identical decisions
for identical inputs. The comparison is done on real semantic bits through the thesis
channels (AWGN and block fading with imperfect CSI):

```
python experiments/grand_verify.py export            # -> results/grand_verify/testset.mat
# then either in MATLAB / Octave:
#   >> cd matlab; addpath('GRAND-MATLAB'); verify_port('../results/grand_verify/testset.mat')
# or, if `matlab` (or `octave-cli`) is on PATH:
python experiments/grand_verify.py run --backend matlab      # or --backend octave
python experiments/grand_verify.py compare                   # expected: 0 mismatches
```

The Python port is run with `budget_mode='level'` for this test because the reference
checks its query budget once per weight level rather than per query (see the docstring
of `grand.orbgrand.ORBGRAND`).

## 3. Use the reference decoder for the evaluation sweeps

```
python experiments/evaluate.py ... --grand_backend matlab     # or octave
```

Every block is decoded by `bin_ORBGRAND1.m` through `orbgrand_decode_file.m` (file
exchange via `.mat`) or through the MATLAB Engine for Python when `matlabengine` is
installed. Expect it to be one to three orders of magnitude slower than the Python
backend (the reference evaluates one noise pattern per loop iteration and is described by
its author as instructional, not optimized), so use it for the final sweeps, with a modest
`--max_queries`, and keep the Python backend for exploration and for training.

## 4. Octave on Colab

```
!apt-get -qq install octave
!bash matlab/fetch_grand_matlab.sh
!python experiments/grand_verify.py export && python experiments/grand_verify.py run --backend octave
```

Duffy's functions only use basic MATLAB (`sort`, `mod`, `cumsum`, `circshift`), which
Octave implements identically.
