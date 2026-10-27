"""
Pluggable ORBGRAND decoding backends.

    fn = make_decode_fn("python" | "matlab" | "octave", H, max_queries, variant)
    words, guesses, abandoned = fn(llr_blocks)          # llr_blocks: (nb, n), positive = bit 0

* python : grand.orbgrand.ORBGRAND (numpy port; fast, used inside training)
* matlab : Ken Duffy's reference code (bin_ORBGRAND1.m / bin_ORBGRAND.m / landslide.m),
           called through the MATLAB Engine for Python if installed, otherwise through
           `matlab -batch` with .mat files for exchange
* octave : the same unmodified .m files run with `octave-cli` (free; `apt install octave`
           on Colab)

The MATLAB/Octave backends need Duffy's files on disk: see matlab/README.md
(`matlab/fetch_grand_matlab.sh` clones https://github.com/kenrduffy/GRAND-MATLAB into
matlab/GRAND-MATLAB, which is gitignored because of its license).

The reference checks its query budget once per weight level; the python backend uses the
same accounting (budget_mode='level', the default here and in the experiment scripts), so
the two backends agree bit for bit (checked by grand_verify.py). Both return 0/1 words with
the hard decisions in abandoned rows (the raw reference returns -1 rows).
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from grand import ORBGRAND  # noqa: E402

MATLAB_DIR = os.path.join(ROOT, "matlab")
DUFFY_DIR = os.path.join(MATLAB_DIR, "GRAND-MATLAB")


def duffy_dirs():
    """Directories that must be on the MATLAB path: ours (drivers) and Duffy's code."""
    dirs = [MATLAB_DIR] + [d for d in (DUFFY_DIR, os.path.join(DUFFY_DIR, "GRAND_Code")) if os.path.isdir(d)]
    # the upstream layout has changed over time: add every folder that holds a bin_ORBGRAND*.m
    for dp, _, fns in os.walk(DUFFY_DIR) if os.path.isdir(DUFFY_DIR) else []:
        if any(f.startswith("bin_ORBGRAND") for f in fns) and dp not in dirs:
            dirs.append(dp)
    return dirs


def _check_duffy():
    for d in duffy_dirs():
        if os.path.exists(os.path.join(d, "bin_ORBGRAND1.m")) and os.path.exists(os.path.join(d, "landslide.m")):
            return
    raise FileNotFoundError("Duffy's GRAND-MATLAB code not found under %s; run matlab/fetch_grand_matlab.sh "
                            "(or clone https://github.com/kenrduffy/GRAND-MATLAB there)" % DUFFY_DIR)


def _matlab_path_cmd():
    return "".join("addpath('%s');" % d.replace("\\", "/").replace("'", "''") for d in duffy_dirs())


def decode_python(llr_blocks, H, max_queries, variant="1line", budget_mode="level"):
    dec = ORBGRAND(H, max_queries=max_queries, variant=variant, budget_mode=budget_mode)
    return dec.decode_many(llr_blocks)


def _clean_words(words, llr_blocks, aband):
    """Reference output -> 0/1 uint8 words with the hard decisions in abandoned rows."""
    hard = (np.asarray(llr_blocks, dtype=np.float64) < 0).astype(np.uint8)
    w = np.rint(np.asarray(words, dtype=np.float64)).astype(np.int64) & 1
    return np.where(aband[:, None], hard, w).astype(np.uint8)


def decode_matlab_files(llr_blocks, H, max_queries, variant="1line", engine="matlab", workdir=None, keep=False):
    """Write llr/H to a .mat, run matlab/orbgrand_decode_file.m with MATLAB or Octave, read back."""
    from scipy.io import loadmat, savemat
    _check_duffy()
    exe = shutil.which("matlab") if engine == "matlab" else (shutil.which("octave-cli") or shutil.which("octave"))
    if exe is None:
        raise FileNotFoundError("%s executable not on PATH" % engine)
    wd = workdir or tempfile.mkdtemp(prefix="grand_%s_" % engine)
    os.makedirs(wd, exist_ok=True)
    fin = os.path.join(wd, "in.mat")
    fout = os.path.join(wd, "out.mat")
    savemat(fin, dict(llr=np.asarray(llr_blocks, dtype=np.float64), H=np.asarray(H, dtype=np.float64),
                      max_queries=float(max_queries), variant=variant))
    code = "%s orbgrand_decode_file('%s','%s');" % (_matlab_path_cmd(), fin.replace("\\", "/"), fout.replace("\\", "/"))
    if engine == "matlab":
        cmd = [exe, "-batch", code]
    else:
        cmd = [exe, "--no-gui", "-q", "--eval", code]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(fout):
        raise RuntimeError("%s failed:\n%s\n%s" % (engine, r.stdout[-2000:], r.stderr[-2000:]))
    out = loadmat(fout)
    guesses = np.rint(np.asarray(out["guesses"], dtype=np.float64)).reshape(-1).astype(np.int64)
    aband = np.rint(np.asarray(out["abandoned"], dtype=np.float64)).reshape(-1).astype(bool)
    words = _clean_words(out["words"], llr_blocks, aband)
    if not keep and workdir is None:
        shutil.rmtree(wd, ignore_errors=True)
    return words, guesses, aband


_ENGINE = None


def decode_matlab_engine(llr_blocks, H, max_queries, variant="1line"):
    """MATLAB Engine for Python (pip install matlabengine, needs a local MATLAB)."""
    global _ENGINE
    import matlab.engine  # noqa: F401
    _check_duffy()
    if _ENGINE is None:
        _ENGINE = matlab.engine.start_matlab()
        for d in duffy_dirs():
            _ENGINE.addpath(d, nargout=0)
    import matlab as ml
    llr = ml.double(np.asarray(llr_blocks, dtype=np.float64).tolist())
    Hm = ml.double(np.asarray(H, dtype=np.float64).tolist())
    words, guesses, aband = _ENGINE.orbgrand_decode_blocks(Hm, float(max_queries), variant, llr, nargout=3)
    aband = np.rint(np.asarray(aband, dtype=np.float64)).reshape(-1).astype(bool)
    guesses = np.rint(np.asarray(guesses, dtype=np.float64)).reshape(-1).astype(np.int64)
    return _clean_words(words, llr_blocks, aband), guesses, aband


def make_decode_fn(backend, H, max_queries, variant="1line", budget_mode="level"):
    if backend == "python":
        return lambda llr: decode_python(llr, H, max_queries, variant, budget_mode)
    if backend == "matlab":
        try:
            import matlab.engine  # noqa: F401
            return lambda llr: decode_matlab_engine(llr, H, max_queries, variant)
        except ImportError:
            return lambda llr: decode_matlab_files(llr, H, max_queries, variant, engine="matlab")
    if backend == "octave":
        return lambda llr: decode_matlab_files(llr, H, max_queries, variant, engine="octave")
    raise ValueError("unknown backend %r" % backend)
