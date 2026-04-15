# Releasing agentis

Publishing uses **PyPI Trusted Publishing** (OIDC). No API tokens are stored as secrets.

## One-time setup

Do this once per registry (TestPyPI and PyPI).

1. Reserve the project name on each registry by creating the project manually or by a first manual upload (`twine upload`). Alternatively, use "pending publishers" — see PyPI docs.
2. On **https://pypi.org/manage/account/publishing/** and **https://test.pypi.org/manage/account/publishing/**, add a trusted publisher with:
   - PyPI project name: `agentis-ai`
   - Owner: `najeeb-thalakkatt`
   - Repository: `agentis`
   - Workflow filename: `publish.yml`
   - Environment name: `pypi` (for PyPI) or `testpypi` (for TestPyPI)
3. In GitHub → Settings → Environments, create two environments: `pypi` and `testpypi`. No secrets needed; the environments exist so the publish jobs can reference them (and so you can add required-reviewer protection later if you want a human gate).

## Cutting a release

```bash
# 1. Bump the version in pyproject.toml (and commit).
#    Use semver: 0.3.0 → 0.3.1 (fix), 0.4.0 (feat), 1.0.0 (major).

# 2. Tag and push.
git tag v0.4.0
git push origin v0.4.0

# 3. Create a GitHub Release from the tag. The `publish.yml` workflow
#    triggers on `release: published` and uploads to PyPI.
gh release create v0.4.0 --generate-notes
```

## Rehearsing on TestPyPI

Before a real release, dry-run end-to-end:

```bash
# Manual dispatch → TestPyPI
gh workflow run publish.yml -f target=testpypi

# Verify the upload installed cleanly:
python -m venv /tmp/agentis-rehearse
/tmp/agentis-rehearse/bin/pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  agentis-ai[anthropic]
/tmp/agentis-rehearse/bin/agentis --version
```

## Checklist

- [ ] CI is green on the commit being released.
- [ ] `pyproject.toml` version bumped and committed.
- [ ] CHANGELOG or release notes ready.
- [ ] Rehearsed on TestPyPI at least once.
- [ ] Tag pushed, GitHub Release published.
- [ ] `pip install agentis-ai==<version>` works from a clean venv.
