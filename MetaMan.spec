# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path.cwd()
logo = project_root / "MetaMan" / "assests" / "metaman.png"
datas = []
if logo.exists():
    datas.append((str(logo), "MetaMan/assests"))

fonts_dir = project_root / "MetaMan" / "assets" / "fonts"
if fonts_dir.is_dir():
    for ttf in fonts_dir.glob("*.ttf"):
        datas.append((str(ttf), "MetaMan/assets/fonts"))


a = Analysis(
    ["run_app.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    # send2trash is imported lazily (recycle-bin deletes) so PyInstaller cannot
    # see it via static analysis; name it explicitly or the frozen build would
    # silently fall back to permanent deletes.
    hiddenimports=["send2trash"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MetaMan",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="MetaMan",
)
