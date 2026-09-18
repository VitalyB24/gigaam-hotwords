"""pyproject.toml declares the runtime dependencies for GitHub's dependency graph; constraints.txt is what the GigaAM
environment is actually installed with. The pins must say the same thing, or the security alerts watch other versions."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pins(lines):
    return {line.split('==')[0].strip(): line.strip() for line in lines if '==' in line and not line.lstrip().startswith('#')}


def test_constraints_are_declared_as_dependencies():
    declared = pins(tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['dependencies'])
    constrained = pins((ROOT / 'constraints.txt').read_text(encoding='utf-8').splitlines())
    assert {'torch', 'torchaudio'} <= set(constrained)
    assert {name: declared.get(name) for name in constrained} == constrained
