import os
import shutil
import subprocess

_copytree = shutil.copytree


def _cow_copytree(src, dst, symlinks=False, ignore=None, copy_function=shutil.copy2,
                  ignore_dangling_symlinks=False, dirs_exist_ok=False):
    """Same result as shutil.copytree for a symlink-free tree, made with APFS
    clones (cp -c); afterwards removes exactly what `ignore` would have skipped."""
    src, dst = os.fspath(src), os.fspath(dst)
    if dirs_exist_ok or os.path.exists(dst) or symlinks:
        return _copytree(src, dst, symlinks=symlinks, ignore=ignore, copy_function=copy_function,
                         ignore_dangling_symlinks=ignore_dangling_symlinks, dirs_exist_ok=dirs_exist_ok)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    subprocess.run(["cp", "-cR", src, dst], check=True)
    if ignore is not None:
        for root, dirs, files in os.walk(src):
            skipped = ignore(root, os.listdir(root))
            rel = os.path.relpath(root, src)
            for name in skipped:
                target = os.path.normpath(os.path.join(dst, rel, name))
                if not target.startswith(os.path.normpath(dst) + os.sep):
                    raise RuntimeError(f"refusing to remove outside the copy: {target}")
                if os.path.isdir(target) and not os.path.islink(target):
                    shutil.rmtree(target)
                elif os.path.lexists(target):
                    os.unlink(target)
            dirs[:] = [d for d in dirs if d not in skipped]
    return dst


shutil.copytree = _cow_copytree
