"""Static import-boundary contracts for the installable package."""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "robotorchan"


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("robotorchan", *parts))


def _resolve_import(
    source: str,
    node: ast.ImportFrom,
    modules: set[str],
    *,
    source_is_package: bool,
) -> set[str]:
    if node.level:
        package = source if source_is_package else source.rpartition(".")[0]
        package_parts = package.split(".")
        keep = len(package_parts) - (node.level - 1)
        prefix = package_parts[:keep]
        if node.module:
            prefix.extend(node.module.split("."))
        base = ".".join(prefix)
    else:
        base = node.module or ""

    targets: set[str] = set()
    for alias in node.names:
        candidate = f"{base}.{alias.name}" if base else alias.name
        if candidate in modules:
            targets.add(candidate)
        elif base in modules:
            targets.add(base)
    return targets


def _import_graph() -> tuple[dict[str, set[str]], set[str]]:
    paths = sorted(SOURCE_ROOT.rglob("*.py"))
    modules = {_module_name(path) for path in paths}
    packages = {
        _module_name(path)
        for path in paths
        if path.name == "__init__.py"
    }
    graph = {module: set() for module in modules}

    for path in paths:
        source = _module_name(path)
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    for size in range(len(parts), 0, -1):
                        target = ".".join(parts[:size])
                        if target in modules:
                            graph[source].add(target)
                            break
            elif isinstance(node, ast.ImportFrom):
                graph[source].update(
                    _resolve_import(
                        source,
                        node,
                        modules,
                        source_is_package=False,
                    )
                )
    return graph, packages


def _find_cycle(graph: dict[str, set[str]]) -> list[str]:
    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(module: str) -> list[str]:
        if module in active_set:
            start = active.index(module)
            return [*active[start:], module]
        if module in visited:
            return []

        visited.add(module)
        active.append(module)
        active_set.add(module)
        for dependency in sorted(graph[module]):
            cycle = visit(dependency)
            if cycle:
                return cycle
        active.pop()
        active_set.remove(module)
        return []

    for module in sorted(graph):
        cycle = visit(module)
        if cycle:
            return cycle
    return []


def test_implementation_modules_do_not_import_aggregator_packages() -> None:
    graph, packages = _import_graph()
    violations = {
        source: sorted(dependencies & packages)
        for source, dependencies in graph.items()
        if dependencies & packages
    }

    assert not violations, (
        "Implementation modules must import owner modules directly instead of "
        f"eager __init__ aggregators: {violations}"
    )


def test_implementation_import_graph_is_acyclic() -> None:
    graph, _ = _import_graph()
    cycle = _find_cycle(graph)

    assert not cycle, f"Internal implementation import cycle detected: {' -> '.join(cycle)}"
