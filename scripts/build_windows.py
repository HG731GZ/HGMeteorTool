"""Build the Windows onedir release.

Select the hgastro interpreter in PyCharm, then run this file directly.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "HoshinoPanoAssistant"

BUILD_DIR = PROJECT_ROOT / "build"
DIST_DIR = PROJECT_ROOT / "dist"
PYINSTALLER_OUTPUT = DIST_DIR / APP_NAME

ARCHITECTURE_NAMES = {
    "amd64": "x64",
    "x86_64": "x64",
    "arm64": "arm64",
    "aarch64": "arm64",
    "x86": "x86",
    "i386": "x86",
    "i686": "x86",
}


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"找不到{description}：{path}")


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"找不到{description}：{path}")


def create_windows_icon(source: Path, target: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "当前 PyCharm 解释器缺少 Pillow；请确认项目使用的是 hgastro 环境。"
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGBA").save(
            target,
            format="ICO",
            sizes=[
                (16, 16),
                (24, 24),
                (32, 32),
                (48, 48),
                (64, 64),
                (128, 128),
                (256, 256),
            ],
        )


def run_pyinstaller(
    icon_path: Path,
    catalog_dir: Path,
    hooks_dir: Path,
) -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "当前 PyCharm 解释器缺少 PyInstaller；请确认项目使用的是 hgastro 环境。"
        ) from exc

    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    environment["PYINSTALLER_CONFIG_DIR"] = str(PROJECT_ROOT / ".pyinstaller-cache")
    environment["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib-cache")

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--contents-directory",
        "_internal",
        "--name",
        APP_NAME,
        "--icon",
        str(icon_path),
        "--additional-hooks-dir",
        str(hooks_dir),
        "--add-data",
        f"{catalog_dir}{os.pathsep}catalog",
        "--add-data",
        f"{PROJECT_ROOT / 'icon256.png'}{os.pathsep}.",
        "--exclude-module",
        "meteoralign.tests",
        "--exclude-module",
        "pytest",
        "--exclude-module",
        "_pytest",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR / "windows"),
        "--specpath",
        str(BUILD_DIR),
        str(PROJECT_ROOT / "main.py"),
    ]

    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def arrange_release(
    release_root: Path,
    release_app_dir: Path,
    metdet_worker_dir: Path,
    qrcode_dir: Path,
) -> None:
    executable = PYINSTALLER_OUTPUT / f"{APP_NAME}.exe"
    require_file(executable, "PyInstaller 生成的主程序")

    if release_root.exists():
        shutil.rmtree(release_root)
    release_root.mkdir(parents=True)

    # 保持和既有 Windows 发布包一致：exe、worker 和 qrcode 位于同一层。
    shutil.move(str(PYINSTALLER_OUTPUT), str(release_app_dir))
    shutil.copytree(metdet_worker_dir, release_app_dir / "metdet_worker")
    shutil.copytree(qrcode_dir, release_app_dir / "qrcode")

def main() -> None:
    if sys.platform != "win32":
        raise RuntimeError("此脚本只能在 Windows 上运行。")

    icon_source = PROJECT_ROOT / "icon256.png"
    icon_output = BUILD_DIR / f"{APP_NAME}.ico"
    catalog_dir = PROJECT_ROOT / "catalog"
    qrcode_dir = PROJECT_ROOT / "qrcode"
    hooks_dir = PROJECT_ROOT / "hooks"

    configured_worker = os.environ.get("METDET_WORKER_DIR", "").strip()
    metdet_worker_dir = (
        Path(configured_worker).expanduser().resolve()
        if configured_worker
        else PROJECT_ROOT / "MetDetPy" / "metdet_worker"
    )

    require_file(icon_source, "应用图标")
    require_directory(catalog_dir, "离线星表目录")
    require_directory(qrcode_dir, "二维码目录")
    require_directory(hooks_dir, "PyInstaller hooks 目录")
    require_directory(metdet_worker_dir, "metdet_worker 目录")
    require_file(metdet_worker_dir / "metdet_worker.exe", "metdet_worker.exe")

    architecture = ARCHITECTURE_NAMES.get(
        platform.machine().lower(), platform.machine().lower() or "unknown"
    )
    release_root = DIST_DIR / f"{APP_NAME}-Win-{architecture}"
    release_app_dir = release_root / APP_NAME

    print(f"使用 Python：{sys.executable}")
    print(f"使用 worker：{metdet_worker_dir}")
    create_windows_icon(icon_source, icon_output)
    run_pyinstaller(icon_output, catalog_dir, hooks_dir)
    arrange_release(release_root, release_app_dir, metdet_worker_dir, qrcode_dir)

    print(f"构建完成：{release_app_dir / f'{APP_NAME}.exe'}")


if __name__ == "__main__":
    main()
