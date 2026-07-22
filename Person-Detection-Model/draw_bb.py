#!/usr/bin/env python3
# =============================================================================
#  draw_boundingbox.py — draw persondetectionmodel output.txt onto the image
#
#  The board writes output.txt with the preprocessing transform on line 1:
#      # transform crop <x0> <y0> <side> of <W0>x<H0>
#      # transform none
#  followed by one detection per line (coords normalized to the 128x128
#  model input): <cx> <cy> <w> <h> [conf]
#
#  This script maps every box back to ORIGINAL image pixels:
#      crop: X = cx*side + x0,  Y = cy*side + y0,  W = w*side,  H = h*side
#      none: X = cx*imgW, ...   (image is used as-is, e.g. .bin inputs)
#
#  Usage:
#      python3 draw_boundingbox.py image.jpg [output.txt]
#              [--out image_boxes.jpg] [--thresh 0.25]
#  Boxes with conf >= --thresh are red; below (top-1 fallback) orange.
# =============================================================================
import argparse
import os
import sys

from PIL import Image, ImageDraw


def parse_output(path):
    transform = ("none",)
    dets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                p = line.split()
                # "# transform crop x0 y0 side of W0xH0" / "# transform none"
                if len(p) >= 3 and p[1] == "transform":
                    if p[2] == "crop" and len(p) >= 6:
                        transform = ("crop", int(p[3]), int(p[4]),
                                     int(p[5]))
                    else:
                        transform = ("none",)
                continue
            p = line.split()
            if len(p) < 4:
                continue
            cx, cy, w, h = map(float, p[:4])
            conf = float(p[4]) if len(p) >= 5 else 1.0
            dets.append((cx, cy, w, h, conf))
    return transform, dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("output", nargs="?", default="output.txt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--thresh", type=float, default=0.25,
                    help="conf below this draws orange (fallback box)")
    args = ap.parse_args()

    transform, dets = parse_output(args.output)
    img = Image.open(args.image).convert("RGB")
    W0, H0 = img.size
    draw = ImageDraw.Draw(img)

    if transform[0] == "crop":
        _, x0, y0, side = transform
        sx = sy = side
        ox, oy = x0, y0
        # sanity: the crop must fit the image it is drawn on
        if x0 + side > W0 or y0 + side > H0:
            print("WARNING: transform (crop %d %d %d) exceeds %dx%d — "
                  "is this the same image the board saw?"
                  % (x0, y0, side, W0, H0))
    else:
        sx, sy, ox, oy = W0, H0, 0, 0

    lw = max(2, round(min(W0, H0) / 300))
    for (cx, cy, w, h, conf) in dets:
        X, Y = cx * sx + ox, cy * sy + oy
        Wb, Hb = w * sx, h * sy
        box = (X - Wb / 2, Y - Hb / 2, X + Wb / 2, Y + Hb / 2)
        color = (255, 0, 0) if conf >= args.thresh else (255, 160, 0)
        draw.rectangle(box, outline=color, width=lw)
        label = "person %.2f" % conf
        tx, ty = box[0], max(0, box[1] - 12 - lw)
        tw = draw.textlength(label) if hasattr(draw, "textlength") \
            else 7 * len(label)
        draw.rectangle((tx, ty, tx + tw + 4, ty + 12 + lw), fill=color)
        draw.text((tx + 2, ty + 1), label, fill=(255, 255, 255))

    out = args.out or (os.path.splitext(args.image)[0] + "_boxes.jpg")
    img.save(out, quality=92)
    print("%d box(es) drawn -> %s" % (len(dets), out))


if __name__ == "__main__":
    main()
