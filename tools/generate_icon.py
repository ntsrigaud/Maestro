"""Generate Maestro application icon assets.

Produces:
  docs/assets/icon.png          — 1024×1024 master PNG
  docs/assets/icon.icns         — macOS icon bundle
  docs/assets/icon.ico          — Windows icon

Run from the repository root:
    python tools/generate_icon.py
"""
from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    sys.exit("Pillow is required: pip install pillow")

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# ── Brand colours ────────────────────────────────────────────────────────────
BG_TOP = (17, 24, 39)       # #111827
BG_BOT = (30, 27, 75)       # #1E1B4B
HAND_TOP = (226, 232, 240)  # #E2E8F0
HAND_BOT = (165, 180, 252)  # #A5B4FC
RING = (129, 140, 248)      # #818CF8
GLOW = (99, 102, 241)       # #6366F1
DOT = (224, 231, 255)       # #E0E7FF


def lerp_colour(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def make_hand_mask(size: int) -> Image.Image:
    """Return an RGBA image with the hand silhouette drawn in white."""
    S = size
    scale = S / 1024.0
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    white = (255, 255, 255, 255)

    def sc(x: float) -> int:
        return int(round(x * scale))

    def rect_rounded(x, y, w, h, r, *, angle=0.0, cx=None, cy=None):
        """Draw a rounded rectangle, optionally rotated around (cx,cy)."""
        if angle == 0.0:
            d.rounded_rectangle([sc(x), sc(y), sc(x + w), sc(y + h)],
                                 radius=sc(r), fill=white)
        else:
            # Render into a temp image then rotate and paste
            margin = int(max(w, h) * scale * 1.5)
            tmp = Image.new("RGBA", (margin * 2, margin * 2), (0, 0, 0, 0))
            td = ImageDraw.Draw(tmp)
            ox, oy = margin - sc(w) // 2, margin - sc(h) // 2
            td.rounded_rectangle([ox, oy, ox + sc(w), oy + sc(h)],
                                  radius=sc(r), fill=white)
            tmp = tmp.rotate(-angle, resample=Image.BICUBIC, expand=False)
            px = int((cx if cx is not None else x + w / 2) * scale) - margin
            py = int((cy if cy is not None else y + h / 2) * scale) - margin
            img.paste(tmp, (px, py), tmp)

    # Palm — extends upward to merge seamlessly with finger bases
    rect_rounded(296, 480, 432, 310, 80)
    # Index finger
    rect_rounded(322, 220, 98, 310, 49)
    # Middle finger (tallest)
    rect_rounded(444, 172, 98, 350, 49)
    # Ring finger
    rect_rounded(566, 220, 98, 310, 49)
    # Pinky
    rect_rounded(680, 278, 86, 260, 43)
    # Thumb (rotated ~20° clockwise)
    rect_rounded(222, 490, 108, 222, 54, angle=20.0, cx=276, cy=601)

    return img


def vertical_gradient_rgba(size: int, y_top: int, y_bot: int,
                            col_top: tuple, col_bot: tuple, alpha: int = 255) -> Image.Image:
    """Return RGBA image filled with a vertical gradient from col_top to col_bot."""
    S = size
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    for y in range(S):
        t = max(0.0, min(1.0, (y - y_top) / max(y_bot - y_top, 1)))
        c = lerp_colour(col_top, col_bot, t)
        for x in range(S):
            img.putpixel((x, y), (*c, alpha))
    return img


def draw_ring(draw: ImageDraw.ImageDraw, cx: float, cy: float,
              rx: float, ry: float, col: tuple, width: int, alpha: int,
              scale: float) -> None:
    x0 = int((cx - rx) * scale)
    y0 = int((cy - ry) * scale)
    x1 = int((cx + rx) * scale)
    y1 = int((cy + ry) * scale)
    draw.ellipse([x0, y0, x1, y1], outline=(*col, alpha), width=max(1, int(width * scale)))


def make_icon(size: int = 1024) -> Image.Image:
    S = size
    scale = S / 1024.0
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── Background: vertical gradient in rounded rect ────────────────
    bg_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(bg_layer)
    # Draw gradient line-by-line
    for y in range(S):
        t = y / S
        c = lerp_colour(BG_TOP, BG_BOT, t)
        bg_draw.line([(0, y), (S - 1, y)], fill=(*c, 255))
    # Clip to rounded rect mask
    mask = Image.new("L", (S, S), 0)
    mask_draw = ImageDraw.Draw(mask)
    r = int(224 * scale)
    mask_draw.rounded_rectangle([0, 0, S - 1, S - 1], radius=r, fill=255)
    img.paste(bg_layer, (0, 0), mask)

    # ── Radial glow behind hand ───────────────────────────────────────
    glow_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    glow_d = ImageDraw.Draw(glow_layer)
    for radius in range(int(340 * scale), 0, -2):
        alpha_val = int(40 * (1 - radius / (340 * scale)))
        cx_ = int(512 * scale)
        cy_ = int(500 * scale)
        glow_d.ellipse(
            [cx_ - radius, cy_ - radius, cx_ + radius, cy_ + radius],
            fill=(*GLOW, alpha_val),
        )
    img = Image.alpha_composite(img, glow_layer)

    # ── Hand gradient fill ────────────────────────────────────────────
    hand_mask = make_hand_mask(S)
    # Build gradient image to colour the hand
    grad_img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    y_top = int(172 * scale)
    y_bot = int(795 * scale)
    for y in range(S):
        t = max(0.0, min(1.0, (y - y_top) / max(y_bot - y_top, 1)))
        c = lerp_colour(HAND_TOP, HAND_BOT, t)
        for x in range(S):
            r_, g_, b_, a_ = hand_mask.getpixel((x, y))
            if a_ > 0:
                grad_img.putpixel((x, y), (*c, a_))
    img = Image.alpha_composite(img, grad_img)

    # ── Activation ring at wrist ──────────────────────────────────────
    draw2 = ImageDraw.Draw(img, "RGBA")
    # Glow ring (wider, semi-transparent)
    draw_ring(draw2, 512, 787, 195, 26, GLOW, 24, 100, scale)
    draw_ring(draw2, 512, 787, 192, 24, GLOW, 20, 70, scale)
    # Crisp main ring
    draw_ring(draw2, 512, 787, 190, 22, RING, 7, 240, scale)
    # Inner highlight
    draw_ring(draw2, 512, 787, 174, 13, (199, 210, 254), 3, 150, scale)

    # ── Baton tip dot above middle finger ────────────────────────────
    cx_d = int(493 * scale)
    cy_d = int(142 * scale)
    r_outer = int(22 * scale)
    r_inner = int(11 * scale)
    # Glow
    for dr in range(r_outer + int(18 * scale), r_outer - 2, -1):
        a = int(60 * max(0, 1 - (dr - r_outer) / (18 * scale)))
        draw2.ellipse(
            [cx_d - dr, cy_d - dr, cx_d + dr, cy_d + dr],
            fill=(*GLOW, a),
        )
    draw2.ellipse([cx_d - r_outer, cy_d - r_outer, cx_d + r_outer, cy_d + r_outer],
                  fill=(*RING, 220))
    draw2.ellipse([cx_d - r_inner, cy_d - r_inner, cx_d + r_inner, cy_d + r_inner],
                  fill=(*DOT, 255))

    return img


def build_icns(png_1024: Path, out: Path) -> None:
    """Build a .icns file from the master 1024×1024 PNG using iconutil."""
    iconset = out.with_suffix(".iconset")
    iconset.mkdir(exist_ok=True)
    master = Image.open(png_1024)
    specs = [
        ("icon_16x16.png", 16),
        ("icon_16x16@2x.png", 32),
        ("icon_32x32.png", 32),
        ("icon_32x32@2x.png", 64),
        ("icon_128x128.png", 128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png", 256),
        ("icon_256x256@2x.png", 512),
        ("icon_512x512.png", 512),
        ("icon_512x512@2x.png", 1024),
    ]
    for name, size in specs:
        resized = master.resize((size, size), Image.LANCZOS)
        resized.save(iconset / name)
    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(out)],
        check=True,
    )
    import shutil
    shutil.rmtree(iconset)
    print(f"  ✓ {out.relative_to(ROOT)}")


def build_ico(png_1024: Path, out: Path) -> None:
    """Build a .ico file with multiple resolutions using Pillow."""
    master = Image.open(png_1024)
    sizes = [16, 32, 48, 64, 128, 256]
    imgs = [master.resize((s, s), Image.LANCZOS) for s in sizes]
    imgs[0].save(out, format="ICO", sizes=[(s, s) for s in sizes],
                 append_images=imgs[1:])
    print(f"  ✓ {out.relative_to(ROOT)}")


def main() -> None:
    print("Generating Maestro icon assets…")

    master_path = ASSETS / "icon.png"
    print("  Rendering 1024×1024 master PNG…")
    icon = make_icon(1024)
    icon.save(master_path, "PNG")
    print(f"  ✓ {master_path.relative_to(ROOT)}")

    if sys.platform == "darwin":
        print("  Building .icns bundle…")
        build_icns(master_path, ASSETS / "icon.icns")

    print("  Building .ico…")
    build_ico(master_path, ASSETS / "icon.ico")

    print("Done.")


if __name__ == "__main__":
    main()
