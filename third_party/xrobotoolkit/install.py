"""Set up the pinned PICO service/SDK and verify the official headset APK."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parent
SERVICE = Path('opt/apps/roboticsservice')


def download_verified(url, archive, expected):
    digest = hashlib.sha256()
    print(f'Downloading {archive.name}', flush=True)
    with urllib.request.urlopen(url, timeout=30) as response, archive.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError('XRoboToolkit download checksum mismatch; nothing was installed')


def download_headset(manifest, root=ROOT):
    headset = manifest['headset']
    apk = root / 'headset' / headset['filename']
    if apk.exists():
        if hashlib.sha256(apk.read_bytes()).hexdigest() != headset['sha256']:
            raise RuntimeError(f'Headset APK checksum mismatch: {apk}; move it aside and download again')
    else:
        apk.parent.mkdir(parents=True, exist_ok=True)
        url = f"{headset['repository']}/releases/download/v{headset['version']}/{headset['filename']}"
        with tempfile.TemporaryDirectory(prefix='.download-', dir=apk.parent) as temporary:
            archive = Path(temporary) / apk.name
            download_verified(url, archive, headset['sha256'])
            archive.rename(apk)
    print(f'Official headset APK verified on this PC: {apk}', flush=True)
    return apk


def _run_adb(adb, device, *arguments):
    result = subprocess.run([adb, '-s', device, *arguments], text=True, capture_output=True, timeout=120)
    if result.returncode:
        detail = (result.stdout + result.stderr).strip()
        if 'INSTALL_FAILED_UPDATE_INCOMPATIBLE' in detail:
            raise RuntimeError('The installed app has a different signing key. Back up its data and remove it '
                               'on the headset before installing the official APK. No app was uninstalled.\n' + detail)
        raise RuntimeError(f'ADB failed for {device}: {detail}')
    return result.stdout.strip()


def verify_headset(manifest, device, adb='adb'):
    headset = manifest['headset']
    paths = _run_adb(adb, device, 'shell', 'pm', 'path', headset['package']).splitlines()
    if len(paths) != 1 or not paths[0].startswith('package:/'):
        raise RuntimeError(f'Expected one installed APK for {headset["package"]}; found {paths}')
    with tempfile.TemporaryDirectory(prefix='egox-headset-check-') as temporary:
        installed = Path(temporary) / 'installed.apk'
        _run_adb(adb, device, 'pull', paths[0].removeprefix('package:'), str(installed))
        digest = hashlib.sha256(installed.read_bytes()).hexdigest()
    if digest != headset['sha256']:
        raise RuntimeError(f'Headset {device} is not running the pinned official APK: SHA256 {digest}. '
                           'A matching version number alone is insufficient.')
    print(f'Installed headset APK verified on {device}: official {headset["version"]}', flush=True)


def install_headset(manifest, device, root=ROOT, adb='adb'):
    apk = download_headset(manifest, root)
    _run_adb(adb, device, 'install', '-r', str(apk))
    verify_headset(manifest, device, adb)


def package_for_host(manifest):
    release = platform.freedesktop_os_release() if sys.platform == 'linux' else {}
    if (release.get('ID') != 'ubuntu' or platform.machine() != 'x86_64'
            or release.get('VERSION_ID') not in manifest['service']['packages']):
        raise RuntimeError('PICO setup supports Ubuntu 22.04/24.04 on x86_64; see third_party/xrobotoolkit/README.md')
    for command in ('dpkg-deb', 'ldd'):
        if shutil.which(command) is None:
            raise RuntimeError(f'PICO setup requires {command}')
    return manifest['service']['packages'][release['VERSION_ID']]


def check_runtime(runtime):
    service = runtime / SERVICE
    executable = service / 'RoboticsServiceProcess'
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f'Incomplete XRoboToolkit runtime: {executable} is missing or not executable')
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = os.pathsep.join(map(str, (service, service / 'lib', service / 'SDK/x64')))
    result = subprocess.run(['ldd', str(executable)], env=env, text=True, capture_output=True, timeout=30)
    if result.returncode or 'not found' in result.stdout:
        raise RuntimeError(f'XRoboToolkit runtime libraries are unavailable:\n{result.stdout}{result.stderr}')


def install_service(manifest, package, root=ROOT):
    runtime = root / 'runtime'
    identity = {'version': manifest['service']['version'], **package}
    marker = runtime / '.egox-install.json'
    if runtime.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise RuntimeError(f'{runtime} already exists without matching installation metadata; '
                               'move it aside before installing this pinned release')
        check_runtime(runtime)
        print(f'XRoboToolkit service already installed: {runtime}', flush=True)
        return runtime

    url = (f"{manifest['service']['repository']}/releases/download/"
           f"v{manifest['service']['version']}/{package['filename']}")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.install-', dir=root) as temporary:
        temporary = Path(temporary)
        archive = temporary / package['filename']
        download_verified(url, archive, package['sha256'])
        extracted = temporary / 'extracted'
        subprocess.run(['dpkg-deb', '--extract', str(archive), str(extracted)], check=True)
        check_runtime(extracted)
        (extracted / '.egox-install.json').write_text(json.dumps(identity, indent=2) + '\n')
        extracted.rename(runtime)
    print(f'XRoboToolkit service installed: {runtime}', flush=True)
    return runtime


def install_sdk(manifest):
    sdk = manifest['python']
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
                    f"{sdk['package']}=={sdk['version']}"], check=True)
    module = importlib.import_module('xrobotoolkit_sdk')
    required = ('init', 'close', 'get_time_stamp_ns', 'get_headset_pose',
                'get_right_controller_pose', 'get_left_controller_pose',
                'get_right_trigger', 'get_left_trigger', 'get_left_axis', 'get_right_axis',
                'get_A_button', 'get_B_button', 'get_X_button', 'get_Y_button', 'get_left_menu_button')
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f'Installed PICO SDK is missing required functions: {missing}')
    print(f"PICO Python SDK {sdk['version']} imported successfully.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='check host support and tools without installing')
    mode.add_argument('--headset-apk', action='store_true', help='only download and verify the official headset APK')
    mode.add_argument('--install-headset', metavar='DEVICE', help='install the official APK on an ADB device and verify it')
    mode.add_argument('--check-headset', metavar='DEVICE', help='verify the installed headset APK without changing it')
    parser.add_argument('--adb', default='adb', metavar='PATH', help='ADB executable (default: adb on PATH)')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if args.headset_apk:
        download_headset(manifest)
        return
    if args.install_headset:
        install_headset(manifest, args.install_headset, adb=args.adb)
        return
    if args.check_headset:
        verify_headset(manifest, args.check_headset, args.adb)
        return
    package = package_for_host(manifest)
    if args.check:
        print(f'PICO setup supported: {package["filename"]}', flush=True)
        return
    install_service(manifest, package)
    install_sdk(manifest)
    download_headset(manifest)
    print('PC setup and APK download complete; the headset app has not been changed.\n'
          'Install on an authorized ADB device with:\n'
          '  python third_party/xrobotoolkit/install.py --install-headset DEVICE', flush=True)
    print('Start tracking with: bash scripts/data_collect_scripts/run_pico_service.sh', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc)) from exc
