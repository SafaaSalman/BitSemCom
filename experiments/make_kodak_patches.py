"""
Make the 24 centre 256x256 crops of the Kodak images that every experiment in this repository
evaluates on (datasets/kodak_patches256). The full images go to datasets/kodak_full first:

  mkdir -p datasets/kodak_full
  for i in $(seq -w 1 24); do wget -q -O datasets/kodak_full/kodim$i.png https://r0k.us/graphics/kodak/kodak/kodim$i.png; done
  python experiments/make_kodak_patches.py
"""
import argparse
import os

from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="datasets/kodak_full")
    ap.add_argument("--dst", default="datasets/kodak_patches256")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    names = sorted(f for f in os.listdir(args.src) if f.lower().endswith(".png"))
    for f in names:
        im = Image.open(os.path.join(args.src, f))
        w, h = im.size
        left, top = (w - args.size) // 2, (h - args.size) // 2
        im.crop((left, top, left + args.size, top + args.size)).save(os.path.join(args.dst, f))
    print("wrote %d patches of %dx%d to %s" % (len(names), args.size, args.size, args.dst))


if __name__ == "__main__":
    main()
