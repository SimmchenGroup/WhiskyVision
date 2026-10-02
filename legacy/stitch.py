import os
import re
import argparse
from collections import defaultdict

GRID_CONFIGS = {
    9:  (3, 3, "Grid: snake by rows", "Right & Down                "),
    16: (4, 4, "Grid: snake by rows", "Right & Down                "),
}

def pairwise_stitch(f1, f2, fused):
    """Single pairwise stitch command between two named images."""
    return (
        f'run("Pairwise stitching", "first_image={f1} second_image={f2} '
        f'fusion_method=[Linear Blending] fused_image={fused} '
        f'check_peaks=5 compute_overlap x=0.0000 y=0.0000 '
        f'registration_channel_image_1=[Average all channels] '
        f'registration_channel_image_2=[Average all channels]");'
    )

def pairwise_block(folder_fwd, out_fwd, prefix, img1, img2, output_name, ext='.tiff'):
    """1x2: open two images, pairwise stitch, save, close all."""
    f1 = f"{prefix}{img1:02d}{ext}"
    f2 = f"{prefix}{img2:02d}{ext}"
    fused = f"{prefix[:-1]}.tiff"

    return [
        f'open("{folder_fwd}/{f1}");',
        f'open("{folder_fwd}/{f2}");',
        f'selectImage("{f2}");',
        pairwise_stitch(f1, f2, fused),
        'run("RGB Color");',
        f'saveAs("Tiff", "{out_fwd}/{output_name}");',
        'close("*");',
        '',
    ]

def pairwise_2x2_block(folder_fwd, out_fwd, prefix, indices, output_name, ext='.tiff'):
    """2x2: stitch 01+02 → row1, stitch 03+04 → row2, stitch row1+row2 → final."""
    f1, f2, f3, f4 = [f"{prefix}{i:02d}{ext}" for i in indices]
    row1 = f"{prefix[:-1]}_row1.tiff"
    row2 = f"{prefix[:-1]}_row2.tiff"
    fused = f"{prefix[:-1]}_fused.tiff"

    return [
        f'// {prefix}  [2x2 double pairwise]',
        # Row 1
        f'open("{folder_fwd}/{f1}");',
        f'open("{folder_fwd}/{f2}");',
        f'selectImage("{f2}");',
        pairwise_stitch(f1, f2, row1),
        f'selectImage("{f1}"); close();',
        f'selectImage("{f2}"); close();',
        '',
        # Row 2 (snake: 04 left, 03 right)
        f'open("{folder_fwd}/{f4}");',
        f'open("{folder_fwd}/{f3}");',
        f'selectImage("{f3}");',
        pairwise_stitch(f4, f3, row2),
        f'selectImage("{f4}"); close();',
        f'selectImage("{f3}"); close();',
        '',
        # Combine rows
        f'selectImage("{row2}");',
        pairwise_stitch(row1, row2, fused),
        f'selectImage("{row1}"); close();',
        f'selectImage("{row2}"); close();',
        'run("RGB Color");',
        f'saveAs("Tiff", "{out_fwd}/{output_name}");',
        'close("*");',
        '',
    ]

def _stitch_row(folder_fwd, prefix, tile_indices, ext='.tiff'):
    """Sequentially pairwise-stitch tile_indices left-to-right; returns (lines, final_image_name)."""
    p = lambda i: f"{prefix}{i:02d}{ext}"
    base = prefix[:-1]
    row_id = f"{tile_indices[0]:02d}_{tile_indices[-1]:02d}"
    prev = f"{base}_row_{row_id}_s1.tiff"
    lines = [
        f'open("{folder_fwd}/{p(tile_indices[0])}");',
        f'open("{folder_fwd}/{p(tile_indices[1])}");',
        f'selectImage("{p(tile_indices[1])}");',
        pairwise_stitch(p(tile_indices[0]), p(tile_indices[1]), prev),
        f'selectImage("{p(tile_indices[0])}"); close();',
        f'selectImage("{p(tile_indices[1])}"); close();',
        '',
    ]
    for step, idx in enumerate(tile_indices[2:], start=2):
        last = (step == len(tile_indices) - 1)
        nxt = f"{base}_row_{row_id}.tiff" if last else f"{base}_row_{row_id}_s{step}.tiff"
        lines += [
            f'open("{folder_fwd}/{p(idx)}");',
            f'selectImage("{p(idx)}");',
            pairwise_stitch(prev, p(idx), nxt),
            f'selectImage("{prev}"); close();',
            f'selectImage("{p(idx)}"); close();',
            '',
        ]
        prev = nxt
    return lines, prev

def pairwise_4x4_block(folder_fwd, out_fwd, prefix, indices, output_name, ext='.tiff'):
    """4x4 snake: build 4 rows via sequential pairwise stitching, then combine rows vertically."""
    row_tiles = [
        list(indices[0:4]),                                     # row 1: L→R  (01-04)
        [indices[7], indices[6], indices[5], indices[4]],      # row 2: spatial L→R (08-05, snake)
        list(indices[8:12]),                                    # row 3: L→R  (09-12)
        [indices[15], indices[14], indices[13], indices[12]],  # row 4: spatial L→R (16-13, snake)
    ]
    lines = [f'// {prefix}  [4x4 pairwise snake]']
    row_names = []
    for tiles in row_tiles:
        row_lines, row_name = _stitch_row(folder_fwd, prefix, tiles, ext=ext)
        lines += row_lines
        row_names.append(row_name)

    base = prefix[:-1]
    r1, r2, r3, r4 = row_names
    rows12  = f"{base}_rows12.tiff"
    rows123 = f"{base}_rows123.tiff"
    fused   = f"{base}_fused.tiff"
    lines += [
        f'selectImage("{r2}");',
        pairwise_stitch(r1, r2, rows12),
        f'selectImage("{r1}"); close();',
        f'selectImage("{r2}"); close();',
        '',
        f'selectImage("{r3}");',
        pairwise_stitch(rows12, r3, rows123),
        f'selectImage("{rows12}"); close();',
        f'selectImage("{r3}"); close();',
        '',
        f'selectImage("{r4}");',
        pairwise_stitch(rows123, r4, fused),
        f'selectImage("{rows123}"); close();',
        f'selectImage("{r4}"); close();',
        'run("RGB Color");',
        f'saveAs("Tiff", "{out_fwd}/{output_name}");',
        'close("*");',
        '',
    ]
    return lines

def grid_block(folder_fwd, out_fwd, prefix, count, output_name, overlap=None, ext='.tiff'):
    """Generate macro lines for Grid/Collection stitching (3x3)."""
    grid_x, grid_y, grid_type, order = GRID_CONFIGS[count]
    overlap_param = f'overlap={overlap} ' if overlap is not None else 'compute_overlap '
    return [
        f"// {prefix}  [{grid_x}x{grid_y}]",
        f'run("Grid/Collection stitching", '
        f'"type=[{grid_type}] '
        f'order=[{order}] '
        f'grid_size_x={grid_x} grid_size_y={grid_y} '
        f'first_file_index_i=1 '
        f'directory=[{folder_fwd}] '
        f'file_names=[{prefix}{{ii}}{ext}] '
        f'output_textfile_name=TileConfiguration.txt '
        f'fusion_method=[Linear Blending] '
        f'regression_threshold=0.10 '
        f'max/avg_displacement_threshold=2.50 '
        f'absolute_displacement_threshold=3.50 '
        f'{overlap_param}'
        f'subpixel_accuracy '
        f'computation_parameters=[Save memory (but be slower)] '
        f'image_output=[Fuse and display]");',
        'run("RGB Color");',
        f'saveAs("Tiff", "{out_fwd}/{output_name}");',
        'close("*");',
        '',
    ]

def generate_macro(folder_path, output_macro, overlap=None):
    pattern = re.compile(
        r'^(S\d+-\d+_4X_[\d.]+uL_'
        r'|(?:large|medium|small)_\w+_\d+ABV_[\d.]+uL_t\d+_'
        r'|(?:large|medium|small)_\d+_[\d.]+uL_'
        r')(\d+)(\.tiff|\.bmp)$',
        re.IGNORECASE,
    )

    file_groups = defaultdict(list)
    for fname in os.listdir(folder_path):
        m = pattern.match(fname)
        if m:
            file_groups[m.group(1)].append((int(m.group(2)), m.group(3).lower()))

    if not file_groups:
        print("No matching files found.")
        return

    folder_fwd = folder_path.replace("\\", "/")
    out_fwd = folder_fwd + "/Stitched"
    lines = [
        f'File.makeDirectory("{out_fwd}");',
        '',
    ]

    for prefix in sorted(file_groups):
        entries = sorted(file_groups[prefix], key=lambda x: x[0])
        indices = [idx for idx, _ in entries]
        ext = entries[0][1]
        count = len(indices)
        output_name = f"{prefix[:-1]}_stitched.tif"

        if count == 2:
            lines += pairwise_block(folder_fwd, out_fwd, prefix, indices[0], indices[1], output_name, ext=ext)
        elif count == 4:
            lines += pairwise_2x2_block(folder_fwd, out_fwd, prefix, indices, output_name, ext=ext)
        elif count == 16:
            lines += pairwise_4x4_block(folder_fwd, out_fwd, prefix, indices, output_name, ext=ext)
        elif count in GRID_CONFIGS:
            lines += grid_block(folder_fwd, out_fwd, prefix, count, output_name, overlap=overlap, ext=ext)
        else:
            print(f"SKIP {prefix}: {count} images (expected 2, 4, 9, or 16)")

    if not lines:
        print("No valid groups to stitch.")
        return

    with open(output_macro, "w") as f:
        f.write("\n".join(lines))

    print(f"Macro written to: {output_macro}\n")
    for prefix in sorted(file_groups):
        count = len(file_groups[prefix])
        layout = {2: "1x2 pairwise", 4: "2x2 double pairwise", 9: "3x3 grid", 16: "4x4 pairwise snake"}.get(count, f"SKIPPED ({count})")
        print(f"  {prefix}  ->  {layout}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an ImageJ macro to stitch renamed tiff tile sets."
    )
    parser.add_argument("folder", help="Folder containing the renamed .tiff files")
    parser.add_argument(
        "--output", default="stitch_macro.ijm",
        help="Output macro filename (default: stitch_macro.ijm)"
    )
    parser.add_argument(
        "--overlap", type=int, default=None, metavar="PCT",
        help="Fixed tile overlap %% for grid stitching (bypasses compute_overlap). "
             "Try 10-20 if auto-detection produces black images."
    )
    args = parser.parse_args()

    output = args.output if args.output != "stitch_macro.ijm" else os.path.join(args.folder, "stitch_macro.ijm")
    generate_macro(args.folder, output, overlap=args.overlap)