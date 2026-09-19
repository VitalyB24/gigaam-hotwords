"""pyproject.toml declares the runtime dependencies for GitHub's dependency graph; constraints.txt is what the GigaAM
environment is actually installed with. The pins must say the same thing, or the security alerts watch other versions.
A package named in both pyproject.toml and requirements-dev.txt must have one range: Dependabot edits the files one by one."""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pins(lines):
    return {line.split('==')[0].strip(): line.strip() for line in lines if '==' in line and not line.lstrip().startswith('#')}


def ranges(lines):
    """Package name -> its requirement without spaces; comments and blank lines are skipped."""
    reqs = [line.strip().replace(' ', '') for line in lines if line.strip() and not line.lstrip().startswith('#')]
    return {re.split(r'[<>=~!]', req, maxsplit=1)[0].lower(): req for req in reqs}


def test_constraints_are_declared_as_dependencies():
    declared = pins(tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['dependencies'])
    constrained = pins((ROOT / 'constraints.txt').read_text(encoding='utf-8').splitlines())
    assert {'torch', 'torchaudio'} <= set(constrained)
    assert {name: declared.get(name) for name in constrained} == constrained


def test_a_package_declared_twice_is_declared_the_same():
    declared = ranges(tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['dependencies'])
    dev = ranges((ROOT / 'requirements-dev.txt').read_text(encoding='utf-8').splitlines())
    both = set(declared) & set(dev)
    assert 'numpy' in both
    assert {name: declared[name] for name in both} == {name: dev[name] for name in both}
