#!/usr/bin/env python3
"""A/B the XY blend fix on real slices, with the real checkpoint.

Runs the same crop through several configurations and reports the tile-seam metric
that flagged the problem on the full prediction (edge density on tile-boundary
columns over edge density elsewhere; ~6x on the original output, 1.0 is clean).

Configurations are run as subprocesses so each imports a different source tree:

  before      the committed branch: top-hat XY blend, guarded normalization,
              zero padding
  after       the fix: Hann XY blend, unconditional normalization, reflect padding
  after-tophat the fixed tree with the blend forced back to top-hat, which isolates
              how much of the change is the blend and how much is the rest

Usage (driver):
  python tools/ab_xy_blend.py --slices 48 --crop 768 --device cuda
Child mode is internal (--child).
"""

import argparse
import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BEFORE_TREE = Path('/home/nmedina/projects/MOSSlite_MOSSfull_merging/merged_moss')
AFTER_TREE = Path(__file__).resolve().parent.parent
INPUT_DIR = Path('/home/nmedina/Documents/Tardi_xy_downsampled_4x/xy')
CHECKPOINT = Path('/home/nmedina/segmentation_projects/little_dots_tardi/subprojects'
                  '/muscles_teresa/checkpoint_unet_3d_slab.pth')
DONE_PRED = Path('/home/nmedina/segmentation_projects/little_dots_tardi/subprojects'
                 '/muscles_teresa/predictions/xy')
ARCH = 'unet_3d_slab'
OVERLAP = 64          # what segmentation_combined_page hardcodes


# ------------------------------------------------------------------------- metrics
def seam_ratio(mask, stride):
    """Edge density on tile-boundary columns vs elsewhere. 1.0 means no tiling."""
    a = mask.astype(np.float32) / 255.0
    gx = np.abs(np.diff(a, axis=1)).mean(axis=0)
    idx = np.arange(len(gx))
    on = gx[idx % stride == (stride - 1) % stride]
    off = gx[idx % stride != (stride - 1) % stride]
    if off.mean() == 0:
        return float('nan')
    return float(on.mean() / off.mean())


def seam_ratio_rows(mask, stride):
    return seam_ratio(mask.T, stride)


def interior(mask, margin=128):
    """Drop a patch-wide border.

    Patches that hang off the edge get padded, and the padding lands in the patch's
    min/max, so it shifts normalization for that patch as a whole rather than only in
    the padded strip. On a small crop those patches are most of the area and would
    dominate any comparison; on the real 3475x5963 volume they are a thin rim. Judging
    the blend on the interior keeps the crop honest about the full-size case.
    """
    if min(mask.shape) <= 2 * margin + 8:
        return mask
    return mask[margin:-margin, margin:-margin]


# --------------------------------------------------------------------------- child
def run_child(args):
    """Predict one configuration. Imports whichever tree was put on sys.path."""
    sys.path.insert(0, args.tree)
    import torch
    from segmentation_suite.models.unet import load_model
    from segmentation_suite.models.slab_inference import read_slab_geometry
    from segmentation_suite.models.architectures import get_3d_patch_size
    from segmentation_suite.workers.predict_worker import PredictWorker

    depth, jitter = read_slab_geometry(str(CHECKPOINT), ARCH)
    patch_size = get_3d_patch_size(ARCH)
    device = torch.device(args.device)
    model = load_model(str(CHECKPOINT), n_channels=1, device=device, architecture=ARCH)

    worker = PredictWorker({})
    worker.log.connect(lambda s: print(f"    [{args.label}] {s}", flush=True))

    kwargs = {}
    if args.xy_blend:            # only the fixed tree accepts this
        kwargs['xy_blend'] = args.xy_blend

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    worker._predict_folder_3d(model, Path(args.crop_dir), out, patch_size, depth,
                              OVERLAP, 'xy', device, z_jitter=jitter, **kwargs)
    print(f"    [{args.label}] done", flush=True)


# -------------------------------------------------------------------------- driver
def pick_crop(crop, z_mid):
    """Choose the densest crop-sized window using predictions already on disk.

    Foreground is ~0.4% of a slice, so a randomly placed crop would likely be empty
    and tell us nothing about seams through real structure.
    """
    preds = sorted(DONE_PRED.glob('*_pred.tif'))
    if not preds:
        raise SystemExit("no finished predictions to locate a data-rich crop from")
    ref = preds[min(z_mid, len(preds) - 1)]
    a = np.array(Image.open(ref)) > 0
    bh, bw = a.shape[0] // crop, a.shape[1] // crop
    if bh == 0 or bw == 0:
        return 0, 0
    trimmed = a[:bh * crop, :bw * crop]
    block = trimmed.reshape(bh, crop, bw, crop).sum(axis=(1, 3))
    by, bx = np.unravel_index(np.argmax(block), block.shape)
    print(f"  crop chosen from {ref.name}: y={by*crop} x={bx*crop} "
          f"(foreground {100*block[by,bx]/crop**2:.2f}% of the window)")
    return by * crop, bx * crop


def build_crop(work, n_slices, crop, z0, y0, x0):
    """Write the cropped sub-volume as a TIFF folder the worker can read."""
    files = sorted(INPUT_DIR.glob('*.tif'))
    sel = files[z0:z0 + n_slices]
    crop_dir = work / 'crop'
    crop_dir.mkdir(parents=True, exist_ok=True)
    for f in sel:
        a = np.array(Image.open(f))
        if a.ndim == 3:
            a = a[..., 0]
        Image.fromarray(a[y0:y0 + crop, x0:x0 + crop]).save(crop_dir / f.name)
    print(f"  crop: {len(sel)} slices of {crop}x{crop} from z={z0}")
    return crop_dir, [f.stem for f in sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--child', action='store_true')
    ap.add_argument('--tree'), ap.add_argument('--out'), ap.add_argument('--crop-dir')
    ap.add_argument('--label', default=''), ap.add_argument('--xy-blend', default='')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--slices', type=int, default=48)
    ap.add_argument('--crop', type=int, default=768)
    ap.add_argument('--z0', type=int, default=2000)
    ap.add_argument('--work', default='/tmp/claude-1001/-home-nmedina/'
                                      'f9fd29e4-629a-442b-ac34-4ef0a60eb584/'
                                      'scratchpad/ab_blend')
    args = ap.parse_args()

    if args.child:
        return run_child(args)

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    print("Building the test crop")
    y0, x0 = pick_crop(args.crop, args.z0)
    crop_dir, stems = build_crop(work, args.slices, args.crop, args.z0, y0, x0)

    configs = [
        ('before',       BEFORE_TREE, ''),
        ('after',        AFTER_TREE,  'hann'),
        ('after-tophat', AFTER_TREE,  'tophat'),
    ]

    results = {}
    for label, tree, blend in configs:
        out = work / f'out_{label}'
        print(f"Running {label} ({tree.name}, blend={blend or 'top-hat (built in)'})")
        cmd = [sys.executable, __file__, '--child', '--tree', str(tree),
               '--out', str(out), '--crop-dir', str(crop_dir), '--label', label,
               '--device', args.device]
        if blend:
            cmd += ['--xy-blend', blend]
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            print(f"  !! {label} failed with code {r.returncode}")
            continue
        masks = np.stack([np.array(Image.open(out / f'{s}_pred.tif')) for s in stems])
        results[label] = masks

    mid = args.slices // 2
    for scope, pick in (('whole crop', lambda m: m), ('interior only', interior)):
        print("\n" + "=" * 74)
        print(f"{scope}")
        print(f"{'config':14s} {'fg %':>7s} {'seam@64':>9s} {'seam@128':>9s} "
              f"{'rows@64':>9s}")
        print("-" * 74)
        for label, masks in results.items():
            m = pick(masks[mid])
            print(f"{label:14s} {100*(m>0).mean():7.3f} "
                  f"{seam_ratio(m, 64):9.2f} {seam_ratio(m, 128):9.2f} "
                  f"{seam_ratio_rows(m, 64):9.2f}")
        print("=" * 74)
    print("seam ratio 1.0 = no tile-boundary edges; the full run measured 5.9 at 64\n")

    for a_label, b_label in (('before', 'after'), ('after-tophat', 'after')):
        if a_label in results and b_label in results:
            a = interior(results[a_label][mid]) > 0
            b = interior(results[b_label][mid]) > 0
            print(f"{a_label} vs {b_label} interior foreground IoU: "
                  f"{(a & b).sum()/max((a | b).sum(), 1):.3f}")
    print("A low IoU means the change altered what is predicted, not only the seams --\n"
          "that needs a human eye on the panel, not just a metric.")

    # Side-by-side panel for eyeballing.
    if results:
        panel = np.concatenate([results[l][mid] for l in results], axis=1)
        png = work / 'compare_mid_slice.png'
        Image.fromarray(panel).save(png)
        print(f"\npanel ({' | '.join(results)}): {png}")
    (work / 'summary.json').write_text(json.dumps(
        {l: {'fg_frac': float((m[mid] > 0).mean()),
             'seam64': seam_ratio(m[mid], 64),
             'seam128': seam_ratio(m[mid], 128)} for l, m in results.items()}, indent=2))


if __name__ == '__main__':
    main()
