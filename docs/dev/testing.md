# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/algorithm.svg" alt="Testing icon"/> Testing

**Audience:** developers who want quick, confidence-building checks before opening a PR.

Lightweight validation strategy:

- Build and run a short CPU test to ensure basic correctness
- Use `RegressionRuns/` to compare outputs between branches
- Verify CLI help (`-h`) matches docs after option changes

## Future work

- Add scripted sanity tests that parse small outputs and compare invariants

## See also

- [Dev → Contributing](contributing.md)
- [Tutorial → First run](../tutorials/first-run.md)
