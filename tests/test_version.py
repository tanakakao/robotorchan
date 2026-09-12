"""Package version tests."""

from importlib.metadata import version

import robotorchan


def test_package_version_matches_installed_metadata() -> None:
    """公開されるpackage versionと``robotorchan.__version__``が一致することを確認する。"""
    assert robotorchan.__version__ == version("robotorchan")
