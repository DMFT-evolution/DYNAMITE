# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/algorithm.svg" alt="Testing icon"/> Testing

Lightweight validation strategy:

- Build and run a short CPU test to ensure basic correctness
- Use `RegressionRuns/` to compare outputs between branches
- Verify CLI help (`-h`) matches docs after option changes

## Future work

- Add scripted sanity tests that parse small outputs and compare invariants
