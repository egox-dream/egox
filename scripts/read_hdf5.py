#!/usr/bin/env python3
"""Inspect HDF5 structure, attributes, masks and storage costs without modifying files.

Adapted from main:policy/read_hdf5_data.py. Raw sizes count dataset payloads,
including encoded JPEG bytes, rather than decoded image sizes.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np


def format_bytes(n_bytes):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n_bytes < 1024:
            return f'{n_bytes:.2f} {unit}'
        n_bytes /= 1024
    return f'{n_bytes:.2f} TB'


def dataset_bytes(dataset):
    """Count variable-length payloads in batches; numeric arrays need no reads."""
    if dataset.dtype.kind != 'O':
        return dataset.size * dataset.dtype.itemsize

    def size(value):
        if isinstance(value, str):
            return len(value.encode('utf-8'))
        if isinstance(value, bytes):
            return len(value)
        return np.asarray(value).nbytes

    if not dataset.shape:
        return size(dataset[()])
    return sum(size(value) for start in range(0, dataset.shape[0], 64)
               for value in dataset[start:start + 64].flat)


def print_hdf5_structure(file_path, indent=0):
    """Print one file and return its on-disk, raw and image-payload byte totals."""
    prefix = '  ' * indent
    try:
        file_size = Path(file_path).stat().st_size
        with h5py.File(file_path, 'r') as root:
            print(f'\n{prefix}File: {file_path}')
            print(f'{prefix}  File size on disk: {format_bytes(file_size)}')
            print(f'{prefix}{"=" * 60}')
            if root.attrs:
                print(f'{prefix}Attributes:')
                for key, value in root.attrs.items():
                    print(f'{prefix}  {key}: {value}')

            total_raw, image_raw = 0, 0

            def print_item(name, obj):
                nonlocal total_raw, image_raw
                item_prefix = '  ' * (indent + 1)
                if isinstance(obj, h5py.Dataset):
                    raw_size = dataset_bytes(obj)
                    total_raw += raw_size
                    if name.startswith('observations/images/'):
                        image_raw += raw_size
                    print(f'{item_prefix}Dataset: {name}')
                    print(f'{item_prefix}  Shape: {obj.shape}  Dtype: {obj.dtype}  raw={format_bytes(raw_size)}')
                    if obj.dtype == bool:
                        print(f'{item_prefix}  Values: {obj[()]}')
                elif isinstance(obj, h5py.Group):
                    print(f'{item_prefix}Group: {name}')

            root.visititems(print_item)
            print(f'\n{prefix}Storage summary:')
            print(f'{prefix}  File size on disk : {format_bytes(file_size)}')
            print(f'{prefix}  Total raw data    : {format_bytes(total_raw)}')
            print(f'{prefix}  Image raw data    : {format_bytes(image_raw)}')
            if total_raw:
                print(f'{prefix}  Image share       : {image_raw / total_raw * 100:.1f}% of raw')
            if file_size:
                print(f'{prefix}  Compression ratio : {total_raw / file_size:.2f}x (raw / on-disk)')
            return file_size, total_raw, image_raw
    except Exception as exc:
        print(f'{prefix}Error reading {file_path}: {exc}')
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', type=Path, help='one HDF5 file or a directory to inspect recursively')
    parser.add_argument('--input_path', type=Path, help='alternative to the positional path')
    args = parser.parse_args(argv)
    if (args.path is None) == (args.input_path is None):
        parser.error('specify one file or directory, using PATH or --input_path PATH')
    path = (args.path or args.input_path).expanduser()
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in ('.hdf5', '.h5'))
    else:
        parser.error(f'path does not exist: {path}')
    if not files:
        print(f'No .hdf5 or .h5 files found in {path}')
        return 1
    print(f'Found {len(files)} HDF5 file(s):')
    summaries = [print_hdf5_structure(file) for file in files]
    readable = [summary for summary in summaries if summary is not None]
    if len(readable) > 1:
        total_file_size, total_raw, total_image_raw = map(sum, zip(*readable))
        print(f'\n{"=" * 60}')
        print(f'Aggregate over {len(readable)} files:')
        print(f'  Total on-disk      : {format_bytes(total_file_size)}')
        print(f'  Total raw data     : {format_bytes(total_raw)}')
        print(f'  Total image raw    : {format_bytes(total_image_raw)}')
        if total_raw:
            print(f'  Image share        : {total_image_raw / total_raw * 100:.1f}% of raw')
        if total_file_size:
            print(f'  Overall compression: {total_raw / total_file_size:.2f}x')
    return int(len(readable) != len(files))


if __name__ == '__main__':
    raise SystemExit(main())
