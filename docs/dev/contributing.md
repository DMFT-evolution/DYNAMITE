# <img class="icon icon-lg icon-primary" src="/DMFE/assets/icons/file.svg" alt="Contributing icon"/> Contributing

Thanks for your interest in contributing to DMFE!

## Workflow

- Fork and create a feature branch
- Keep changes focused; add or update docs when public behavior changes
- Open a PR with a clear summary and testing evidence

## Code style

- C++17, consistent formatting (clang-format if config exists)
- Doxygen comments for public APIs
- Keep lines ≤ 100 chars; prefer small, focused headers and source files

## Testing

- Perform a short run (CPU is fine) and include key metrics/logs
- For changes to outputs, document the effect in `docs/usage.md`

## Docs

- Update or add pages under `docs/` (how-tos, concepts, tutorials)
- Run `./docs/gen_api.sh` if you changed public headers
