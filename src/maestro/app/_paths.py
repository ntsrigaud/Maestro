"""Runtime path resolution for the Maestro configuration directory.

Resolution priority (first match wins):

1. ``MAESTRO_CONFIG_DIR`` environment variable — CI pipelines, containers,
   and developer overrides.  Set this to any directory that contains the
   canonical ``app.yaml`` / ``inference.yaml`` / … tree.

2. PyInstaller frozen bundle — when the application is packaged as a
   standalone binary, ``sys._MEIPASS`` contains a ``configs/`` subtree
   extracted from the archive.

3. User config directory — the OS-appropriate per-user location:
   - macOS   : ``~/Library/Application Support/Maestro/configs/``
   - Windows  : ``%APPDATA%\\Maestro\\configs\\``
   - Linux/XDG: ``$XDG_CONFIG_HOME/Maestro/configs/`` (default
                  ``~/.config/Maestro/configs/``)
   On first launch this directory is populated automatically from the
   bundled defaults shipped with the package.

4. ``configs/`` relative to the current working directory — development
   convenience; works when the repository root is the working directory.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _user_config_dir() -> Path:
    """Return the OS-appropriate per-user Maestro data directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:  # Linux / XDG
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "Maestro"


def _bundled_defaults_dir() -> Path | None:
    """Return the path to the bundled default configs inside the installed package.

    Returns ``None`` when the package is not installed with its data files
    (e.g. inside a PyInstaller bundle that handles data files separately).
    """
    try:
        import maestro._defaults as _mod  # type: ignore[import]

        if _mod.__file__ is None:
            return None
        return Path(_mod.__file__).parent
    except (ModuleNotFoundError, AttributeError, TypeError):
        return None


def _copy_defaults_to_user_dir(user_dir: Path) -> None:
    """Populate *user_dir/configs* from the bundled package defaults.

    Called exactly once on first launch when no existing config tree is found.
    Raises ``RuntimeError`` if the bundled defaults cannot be located (which
    should never happen in a correctly installed package).
    """
    src = _bundled_defaults_dir()
    if src is None:
        raise RuntimeError(
            "Maestro could not locate the bundled default configuration files.\n"
            "Please reinstall the package, or set the MAESTRO_CONFIG_DIR environment\n"
            "variable to a directory containing a valid Maestro configuration tree."
        )
    dest = user_dir / "configs"
    LOGGER.info("First launch: copying default configs from %s to %s", src / "configs", dest)
    user_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "configs", dest)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_config_dir() -> Path:
    """Resolve and return the absolute path to the Maestro configuration directory.

    See module docstring for the full resolution order.
    """
    # 1. Explicit env-var override
    env_override = os.environ.get("MAESTRO_CONFIG_DIR")
    if env_override:
        path = Path(env_override).expanduser().resolve()
        LOGGER.debug("Config dir from MAESTRO_CONFIG_DIR: %s", path)
        return path

    # 2. PyInstaller frozen bundle
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        path = Path(sys._MEIPASS) / "configs"  # type: ignore[attr-defined]
        LOGGER.debug("Config dir from PyInstaller bundle: %s", path)
        return path

    # 3a. User config directory already populated
    user_configs = _user_config_dir() / "configs"
    if user_configs.exists():
        LOGGER.debug("Config dir from user config directory: %s", user_configs)
        return user_configs

    # 4. Development fallback: CWD/configs
    cwd_configs = Path.cwd() / "configs"
    if cwd_configs.exists():
        LOGGER.debug("Config dir from CWD (dev mode): %s", cwd_configs)
        return cwd_configs

    # 3b. First launch: user config directory does not exist yet — populate it
    _copy_defaults_to_user_dir(_user_config_dir())
    LOGGER.info("Config dir initialised at: %s", user_configs)
    return user_configs
