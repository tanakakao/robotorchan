"""Documentation consistency contracts for the acquisition registry."""

from pathlib import Path

from robotorchan.acquisition.registry import ACQUISITION_REGISTRY

ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_GUIDE = (
    ROOT / "docs-internal" / "contracts" / "optimization" / "acquisition-integration.md"
)
THEORY_ROOT = ROOT / "docs" / "theory" / "acquisition"
GUIDE_ROOT = ROOT / "docs" / "guide" / "acquisition"


def _documentation_text() -> str:
    guide = "\n".join(path.read_text(encoding="utf-8") for path in sorted(GUIDE_ROOT.glob("*.md")))
    integration = INTEGRATION_GUIDE.read_text(encoding="utf-8")
    return f"{integration}\n{guide}"


def test_registered_acquisitions_are_named_in_documentation() -> None:
    """Every registry acquisition must be discoverable in current documentation."""
    documentation = _documentation_text()

    missing = sorted(name for name in ACQUISITION_REGISTRY if name not in documentation)

    assert missing == [], f"Undocumented acquisition registry entries: {missing}"


def test_acquisition_documentation_sources_exist() -> None:
    """Registry documentation contracts must point to durable documentation roots."""
    assert INTEGRATION_GUIDE.is_file()
    assert THEORY_ROOT.is_dir()
    assert GUIDE_ROOT.is_dir()
    assert any(GUIDE_ROOT.glob("*.md"))
    assert any(THEORY_ROOT.glob("*.md"))
