#!/usr/bin/env python3
# Decoder for the OV5642 (5MP) capture path. makeimg.py stays as-is for the
# 2MP OV2640. Two differences from makeimg.py, both verified against real
# captures from this board:
#   1. camcap5's image.bin carries a few stale FIFO bytes up front
#      (153608 instead of 153600) -> the frame is the LAST 153600 bytes.
#   2. The OV5642 emits UYVY byte order (U Y0 V Y1), not the OV2640's
#      order. Decoding with U/V swapped renders red fabric as blue.
#
# Usage:
#   python3 makeimg5.py                      # image.bin -> image5.jpg
#   python3 makeimg5.py mycap.bin -o out.jpg
#   python3 makeimg5.py --enhance            # + unsharp mask + 2x upscale
#   python3 makeimg5.py --awb                # + gray-world white balance
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

WIDTH = 320
HEIGHT = 240
FRAME = WIDTH * HEIGHT * 2


def decode_uyvy(data):
    arr = np.frombuffer(data, dtype=np.uint8).reshape((HEIGHT, WIDTH // 2, 4))

    u, y0, v, y1 = (arr[:, :, i].astype(np.float32) for i in range(4))
    u -= 128
    v -= 128

    def convert(y):
        r = y + 1.402 * v
        g = y - 0.34414 * u - 0.71414 * v
        b = y + 1.772 * u
        return np.dstack((r, g, b))

    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    rgb[:, 0::2, :] = np.clip(convert(y0), 0, 255)
    rgb[:, 1::2, :] = np.clip(convert(y1), 0, 255)
    return rgb


def gray_world_awb(rgb):
    result = rgb.astype(np.float32)
    means = result.reshape(-1, 3).mean(axis=0)
    target = means.mean()
    for c in range(3):
        if means[c] > 0:
            result[:, :, c] *= target / means[c]
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description="OV5642 YUV422 (UYVY) -> JPEG")
    p.add_argument("input", nargs="?", default="image.bin")
    p.add_argument("-o", "--output", default="image5.jpg")
    p.add_argument("--awb", action="store_true",
                   help="gray-world white balance (sensor AWB is usually enough)")
    p.add_argument("--enhance", action="store_true",
                   help="unsharp mask + 2x Lanczos upscale")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] {args.input} not found.")
        sys.exit(1)

    with open(args.input, "rb") as f:
        raw = f.read()
    print(f"Loaded {args.input}: {len(raw)} bytes")

    # camcap5's jpeg mode already writes a finished .jpg; handle one anyway
    # in case its raw FIFO dump gets passed through here.
    soi = raw.find(b"\xff\xd8")
    if 0 <= soi < 64:
        eoi = raw.find(b"\xff\xd9", soi)
        if eoi > 0:
            with open(args.output, "wb") as f:
                f.write(raw[soi:eoi + 2])
            print(f"[SUCCESS] Input was already JPEG; trimmed to {args.output}")
            return

    if len(raw) < FRAME:
        print(f"[WARN] Short file (expected >= {FRAME}), zero-padding.")
        raw += b"\x00" * (FRAME - len(raw))
    offset = len(raw) - FRAME
    if offset:
        print(f"Skipping {offset} stale FIFO byte(s) at the start.")

    rgb = decode_uyvy(raw[offset:offset + FRAME])
    if args.awb:
        print("Applying gray-world white balance...")
        rgb = gray_world_awb(rgb)

    img = Image.fromarray(rgb, "RGB")
    if args.enhance:
        print("Enhancing: 2x Lanczos upscale + unsharp mask...")
        img = img.resize((WIDTH * 2, HEIGHT * 2), Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))

    img.save(args.output, quality=95)
    print(f"[SUCCESS] Saved {args.output} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
