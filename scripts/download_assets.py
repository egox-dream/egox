"""Download Unitree's Isaac Lab asset archive and extract the G1 Dex3 robot USD into assets/."""
import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

URL = ('https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds/resolve/'
       '394cf2448f8a9ed815c77c701a761f3d1ff1c8fb/assets.zip')
SHA256 = '06fbf14549be3a81e3dbd4a8a019e7f060f48de1556b70f291a18ac8f9ead4b0'  # archive version used for the dataset
PREFIX = 'assets/robots/g1-29dof_wholebody_dex3/'
ARCHIVE = Path.home() / '.cache/egox_sim/unitree_assets.zip'
DESTINATION = Path(__file__).resolve().parents[1] / 'assets/robots/g1_dex3/usd'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 23), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if not ARCHIVE.exists():
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        partial = ARCHIVE.with_suffix('.part')
        print(f'Downloading {URL} (1.3 GB)', flush=True)
        with urllib.request.urlopen(URL, timeout=60) as source, partial.open('wb') as target:
            shutil.copyfileobj(source, target)
        partial.rename(ARCHIVE)
    if sha256(ARCHIVE) != SHA256:
        raise ValueError(f'{ARCHIVE} differs from the pinned archive; delete it to download again')
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = [m for m in archive.infolist()
                   if m.filename.startswith(PREFIX) and not m.is_dir() and '..' not in m.filename]
        for member in members:
            target = DESTINATION / member.filename[len(PREFIX):]
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open('wb') as stream:
                shutil.copyfileobj(source, stream)
    print(f'Extracted {len(members)} files to {DESTINATION}')


if __name__ == '__main__':
    main()
