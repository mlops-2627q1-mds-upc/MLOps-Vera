<!-- Title: Conventional Commits format, e.g. "feat(data): add caption-grouped split stage" (see CONTRIBUTING.md) -->

## Why

Jira: KAN-

<!-- Which problem does this solve, or which requirement does it implement? -->

## What is changing

-

## How to test

<!-- Commands a reviewer can run, e.g. uv sync, uv run pytest, uv run dvc repro -->

## Checklist

- [ ] The title follows Conventional Commits
- [ ] `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest` pass
- [ ] No data or credentials are committed (`data/`, `.dvc/config.local`, `.env`)
- [ ] If the data changed: `dvc.lock` is updated and `dvc push` is done
- [ ] README, cards or report are updated if needed
- [ ] The Jira ticket is linked and moved to "En revisión"
