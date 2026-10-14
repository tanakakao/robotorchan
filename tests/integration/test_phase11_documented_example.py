"""Keep the public unified API example runnable as documented."""

import runpy
from pathlib import Path


def test_documented_unified_api_example_runs_end_to_end() -> None:
    """Validate the published example against actual model and optimizer APIs."""
    example = Path(__file__).resolve().parents[2] / "examples" / "unified_api_workflow.py"
    runpy.run_path(str(example), run_name="__main__")
