# Audit execution record

Source revision: `bc7cd2b04425220965bc732b8b80ce54da450375`.

Selected existing tests: `tests/test_contracts.py`, `tests/test_semantic_validator.py`, `tests/test_providers_unit.py`.

Result: **19 passed, 1 deselected in 2.09 seconds**. The configuration test requiring a secret/proxy was deselected. Tests requiring live provider calls were not run. Pytest automatic third-party plugin loading was disabled, and pytest-asyncio was explicitly enabled. A Windows registry lookup was suppressed because it is unavailable to the desktop runner. Local asyncio event-loop creation required execution outside the restricted runner; no live model API was called.

Custom probes: **15 groups completed without harness errors**. See `reproduction-results.json` for expected adversarial failures. These are demonstrations that the audited implementation accepts bad conditions, not tests asserting the implementation is correct. The harness redirects source fixture paths, mocks remote SAM responses, and writes generated/corrupted CAD files only under `cad-probe/`.

Proposed helpers: synthetic height-plane projection recovered world coordinates within 3.64e-12 mm; fast async completion returned `complete`; delayed completion returned `unknown_timeout`. These do not establish real-camera accuracy or a production latency guarantee. CadQuery was unavailable, so the optional CAD helper was not executed.

Dependency note: the isolated target installation supplied `meta-sam-parser==0.1.0`, pytest, pytest-asyncio, trimesh, shapely and reportlab. Its NumPy version produced a compatibility warning with the existing SciPy installation. No global dependency package was changed. The original repository has no pinned Python dependency manifest; the audit is not a reproducible release environment certification.

The original checkout remained clean after execution. No application changes, commits, deployment changes, PRs, or external messages were made.
