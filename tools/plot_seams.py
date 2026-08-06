#!/usr/bin/env python3
"""Plots for the XY tiling fix, from an ab_xy_blend.py run directory.

Writes into ~/moss_tiling_plots:
  1_seam_profile.png    edge density per column, one slice, tile boundaries marked
  2_seam_per_slice.png  seam ratio for every slice, before vs after
  3_slice_<z>.png       raw | before | after | disagreement, full slice + zoom
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BEFORE, AFTER = '#D1495B', '#1C7ED6'      # validated categorical pair
INK, MUTED, GRID = '#22252a', '#6b7280', '#dcdfe4'
STRIDE = 64                                # patch 128 - overlap 64


def edge_profile(mask):
    """Mean |horizontal gradient| per column: where the mask has vertical edges."""
    return np.abs(np.diff(mask.astype(np.float32) / 255.0, axis=1)).mean(axis=0)


def seam_ratio(mask, stride=STRIDE):
    g = edge_profile(mask)
    i = np.arange(len(g))
    on, off = g[i % stride == stride - 1], g[i % stride != stride - 1]
    return float(on.mean() / off.mean()) if off.mean() else np.nan


def interior(m, margin=128):
    return m[margin:-margin, margin:-margin] if min(m.shape) > 2 * margin + 8 else m


def style(ax):
    ax.set_facecolor('#fcfcfb')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def plot_phase(bef, aft, out):
    """Edge density folded onto position-within-tile.

    A single slice's raw profile is dominated by real object boundaries, which sit
    wherever the biology is and drown the seam out. Folding every column onto its
    offset within the 64 px tile and averaging over all slices cancels that: only a
    tile-locked artefact survives, and it lands on the last offset.
    """
    def fold(stack):
        g = np.stack([edge_profile(interior(m)) for m in stack])   # (z, cols)
        ph = np.arange(g.shape[1]) % STRIDE
        return np.array([g[:, ph == p].mean() for p in range(STRIDE)])

    fb, fa = fold(bef), fold(aft)
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    style(ax)
    xs = np.arange(STRIDE)
    ax.axvline(STRIDE - 1, color=MUTED, lw=0.9, ls=':', alpha=0.8)
    ax.annotate('tile boundary', (STRIDE - 1, max(fb.max(), fa.max())),
                xytext=(-8, -4), textcoords='offset points', ha='right',
                color=MUTED, fontsize=9)
    ax.plot(xs, fb, color=BEFORE, lw=2, label='before (top-hat blend)')
    ax.plot(xs, fa, color=AFTER, lw=2, label='after (Hann taper)')
    ax.annotate(f'{fb[-1]/np.median(fb):.1f}x median', (xs[-1], fb[-1]),
                xytext=(-10, 10), textcoords='offset points', ha='right',
                color=BEFORE, fontsize=10)
    ax.set_xlabel(f'offset within {STRIDE} px tile (px)', color=INK, fontsize=10)
    ax.set_ylabel('mean edge density', color=INK, fontsize=10)
    ax.set_title(f'Edge density vs position within the tile — averaged over '
                 f'{len(bef)} slices\nonly a tile-locked artefact survives this fold',
                 color=INK, fontsize=12, loc='left')
    ax.set_ylim(0, None)
    leg = ax.legend(frameon=False, fontsize=9, loc='upper left')
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out, facecolor='#fcfcfb')
    plt.close(fig)


def plot_per_slice(bef, aft, out):
    rb = np.array([seam_ratio(interior(m)) for m in bef])
    ra = np.array([seam_ratio(interior(m)) for m in aft])
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    style(ax)
    zs = np.arange(len(rb))
    ax.axhline(1.0, color=MUTED, lw=1.2, ls='--', alpha=0.8)
    ax.annotate('1.0 = no tile seams', (2, 1.0), xytext=(6, 8),
                textcoords='offset points', ha='left', color=MUTED, fontsize=9)
    ax.plot(zs, rb, color=BEFORE, lw=2, label='before (top-hat blend)')
    ax.plot(zs, ra, color=AFTER, lw=2, label='after (Hann taper)')
    ax.annotate(f'mean {np.nanmean(rb):.1f}x', (zs[-1], rb[-1]), xytext=(8, 0),
                textcoords='offset points', color=BEFORE, fontsize=10, va='center')
    ax.annotate(f'mean {np.nanmean(ra):.1f}x', (zs[-1], ra[-1]), xytext=(8, 0),
                textcoords='offset points', color=AFTER, fontsize=10, va='center')
    ax.set_xlabel('slice index in test crop', color=INK, fontsize=10)
    ax.set_ylabel('seam ratio', color=INK, fontsize=10)
    ax.set_title('Tile-seam strength per slice\n'
                 'edge density on tile boundaries / edge density elsewhere',
                 color=INK, fontsize=12, loc='left')
    ax.set_xlim(0, len(rb) + 4)
    ax.set_ylim(0, max(2.0, np.nanmax(rb) * 1.1))
    leg = ax.legend(frameon=False, fontsize=9, loc='upper left')
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out, facecolor='#fcfcfb')
    plt.close(fig)


def plot_slice(raw, bef, aft, out, z, zs=288):
    b, a = bef > 0, aft > 0
    # Zoom where the two disagree most -- that is where a seam was.
    best = (-1, 0, 0)
    for yy in range(0, raw.shape[0] - zs, STRIDE):
        for xx in range(0, raw.shape[1] - zs, STRIDE):
            s = (b[yy:yy + zs, xx:xx + zs] ^ a[yy:yy + zs, xx:xx + zs]).sum()
            if s > best[0]:
                best = (s, yy, xx)
    _, y0, x0 = best

    fig, axes = plt.subplots(2, 4, figsize=(15, 8), dpi=140)
    for row, (sl, tag) in enumerate([((slice(None), slice(None)), 'full crop'),
                                     ((slice(y0, y0 + zs), slice(x0, x0 + zs)),
                                      f'zoom y={y0} x={x0}')]):
        r, bb, aa = raw[sl], b[sl], a[sl]
        for ax in axes[row]:
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
        axes[row][0].imshow(r, cmap='gray')
        axes[row][0].set_title(f'raw EM — {tag}', fontsize=10, color=INK, loc='left')
        for col, (m, c, nm) in enumerate([(bb, BEFORE, 'before'), (aa, AFTER, 'after')], 1):
            axes[row][col].imshow(r, cmap='gray')
            ov = np.zeros(m.shape + (4,))
            ov[m] = list(matplotlib.colors.to_rgb(c)) + [0.65]
            axes[row][col].imshow(ov)
            axes[row][col].set_title(nm, fontsize=10, color=c, loc='left')
        d = np.zeros(bb.shape + (3,))
        d[bb & aa] = (0.27, 0.28, 0.30)
        d[bb & ~aa] = matplotlib.colors.to_rgb(BEFORE)
        d[aa & ~bb] = matplotlib.colors.to_rgb(AFTER)
        axes[row][3].imshow(d)
        axes[row][3].set_title('disagreement', fontsize=10, color=INK, loc='left')
        if row == 1:                       # mark the tile grid on the zoom
            for col in range(4):
                for gx in range(STRIDE - (x0 % STRIDE), zs, STRIDE):
                    axes[row][col].axvline(gx, color='#ffd166', lw=0.8, alpha=0.75)
    fig.suptitle(f'Slice z={z}: grey = both agree, red = before only, blue = after only.'
                 '  Yellow lines on the zoom are tile boundaries.',
                 fontsize=11, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.subplots_adjust(hspace=0.14)      # room for the zoom row's own titles
    fig.savefig(out, facecolor='#fcfcfb')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='/tmp/claude-1001/-home-nmedina/'
                                     'f9fd29e4-629a-442b-ac34-4ef0a60eb584/'
                                     'scratchpad/ab_gpu')
    ap.add_argument('--out', default=str(Path.home() / 'moss_tiling_plots'))
    ap.add_argument('--slices', type=int, nargs='*', default=None)
    args = ap.parse_args()

    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in (run / 'crop').glob('*.tif'))
    load = lambda d, s: np.array(Image.open(run / d / f'{s}_pred.tif'))
    bef = np.stack([load('out_before', s) for s in stems])
    aft = np.stack([load('out_after', s) for s in stems])
    raws = np.stack([np.array(Image.open(run / 'crop' / f'{s}.tif')) for s in stems])

    mid = len(stems) // 2
    plot_phase(bef, aft, out / "1_seam_phase.png")
    plot_per_slice(bef, aft, out / '2_seam_per_slice.png')
    for z in (args.slices or [mid, mid + 12]):
        plot_slice(raws[z], bef[z], aft[z], out / f'3_slice_{z:03d}.png', z)

    print(f"wrote {len(list(out.glob('*.png')))} plots to {out}")
    for p in sorted(out.glob('*.png')):
        print("  ", p)


if __name__ == '__main__':
    main()
