"""Download EgoX HDF5 episodes from Hugging Face for local training."""
import argparse
from collections import Counter
from pathlib import Path

REPO_ID = 'binghong0727/egox'
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'data'
EMBODIMENTS = ('g1-dex5', 'g1-sim-dex3', 'human-dex5', 'locoman-dex2',
               'r1lite-gripper', 'xarm7-dex4', 'z1-tool')


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embodiment', type=str.lower, choices=('all', *EMBODIMENTS), default='all',
                        help='download only this embodiment (case-insensitive; default: all)')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT,
                        help='dataset root (default: data/ under the repository, independent of working directory)')
    parser.add_argument('--revision', default='main', help='Hub branch, tag, or commit (default: main)')
    parser.add_argument('--max-workers', type=positive_int, default=4,
                        help='concurrent file downloads (default: 4)')
    parser.add_argument('--dry-run', action='store_true',
                        help='list selected file counts and total size without downloading episodes')
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, RepoFile, snapshot_download
    except ImportError:
        parser.exit(1, 'Install the downloader dependency with: python -m pip install "huggingface_hub>=1.0"\n')

    output = args.output_dir.expanduser().resolve()
    api = HfApi()
    # Resolve once so the inventory and downloads refer to exactly the same version.
    revision = api.dataset_info(REPO_ID, revision=args.revision).sha
    folder = None if args.embodiment == 'all' else args.embodiment
    files = [entry for entry in api.list_repo_tree(REPO_ID, repo_type='dataset', revision=revision,
                                                  path_in_repo=folder, recursive=True)
             if isinstance(entry, RepoFile) and entry.path.endswith('.hdf5')]
    if not files:
        parser.exit(1, f'No HDF5 episodes found for {args.embodiment} at {revision}.\n')
    counts = Counter(entry.path.split('/')[0] for entry in files)
    size = sum(entry.size for entry in files)
    print(f'Dataset: https://huggingface.co/datasets/{REPO_ID}', flush=True)
    print(f'Revision: {revision}', flush=True)
    print(f'Destination: {output}', flush=True)
    for embodiment, count in sorted(counts.items()):
        print(f'  {embodiment}: {count:,} HDF5 files', flush=True)
    print(f'Selected: {len(files):,} files, {size / 1e9:,.2f} GB ({size / 2**30:,.2f} GiB) total', flush=True)
    if args.dry_run:
        print('Dry run: no episodes downloaded. Total size includes files you may already have.')
        return

    # Wildcards include nested recording folders. Keep Hub paths and local download
    # metadata intact so rerunning can reuse completed files and resume partial files.
    pattern = '*.hdf5' if folder is None else f'{folder}/*.hdf5'
    snapshot_download(repo_id=REPO_ID, repo_type='dataset', revision=revision, local_dir=output,
                      allow_patterns=pattern, max_workers=args.max_workers)
    print(f'HDF5 episodes ready in {output}', flush=True)


if __name__ == '__main__':
    main()
