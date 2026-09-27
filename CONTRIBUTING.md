# Contributing to MLOps-Vera

How we work on this repository. Code is versioned with Git using GitHub Flow, data with DVC (remote storage on DagsHub), and tasks are planned on the team's Jira board (project `KAN`).

## Workflow

1. Take a ticket on the Jira board and move it to **En curso**.
2. Start from an up-to-date `main` and create a branch for the ticket:

   ```bash
   git switch main
   git pull
   git switch -c feat/KAN-12-mlflow-tracking
   ```

3. Commit small, related changes using [Conventional Commits](#commit-messages), and push the branch with `git push -u origin <branch>`.
4. Open a pull request and fill in the template (why, what is changing, how to test). Move the ticket to **En revisión**.
5. A teammate reviews the pull request. Merge it once it has one approval, then delete the branch.
6. Move the ticket to **Finalizado** and add a comment with the pull request link.

Never push directly to `main`, never rewrite history that is already on GitHub (no force pushes), and never commit secrets (see [Data and credentials](#data-and-credentials)).

## Branch names

`<type>/<JIRA-KEY>-<short-description>`, using the same types as commit messages:

- `feat/KAN-12-mlflow-tracking`
- `fix/KAN-18-caption-split`
- `docs/KAN-20-retrospective`

## Commit messages

We follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/):

```text
<type>(<optional scope>): <description>

<optional body>

<optional footer(s)>
```

| Type | Use it for |
| --- | --- |
| `feat` | New functionality: a pipeline stage, a training script, an API endpoint |
| `fix` | A bug fix |
| `docs` | Documentation only: README, cards, report |
| `test` | Adding or correcting tests |
| `refactor` | A code change that neither fixes a bug nor adds a feature |
| `perf` | A performance improvement |
| `style` | Formatting only, no change in behaviour |
| `build` | Dependencies and packaging: `pyproject.toml`, `uv.lock`, Docker |
| `ci` | CI configuration: GitHub Actions workflows |
| `chore` | Other maintenance, such as `.gitignore` or tool settings |
| `revert` | Reverting a previous commit |

The scope is optional and names the part of the project that changes, for example `data`, `features`, `model`, `api`, `dvc`, `report`, `cards` or `deps`.

Rules:

- Write the description in the imperative mood, starting in lowercase, with no final period: `feat(data): add caption-grouped split stage`.
- Keep the header (type, scope and description) under 72 characters, and aim for about 50.
- Leave a blank line after the header. Wrap the body at 72 characters and use it to explain what changed and why, not how.
- Reference the Jira ticket in a footer: `Refs: KAN-12`.
- Mark a breaking change with `!` after the type or scope, and describe it in a `BREAKING CHANGE:` footer.

These rules keep the course's commit message guidelines: a blank line before the body, the imperative mood, no final period, a body wrapped at 72 characters that explains what and why. The only adaptations come from the type prefix: a longer header limit, and a lowercase description as in the Conventional Commits examples.

Example:

```text
feat(data): add caption-grouped train/val/test split

Images generated from the same caption must stay in the same split,
otherwise near-duplicates leak from training into evaluation. The
split stage groups by caption and fails if any caption leaks.

Refs: KAN-18
```

## Pull requests

- Use the same format for the title, for example `feat(model): add baseline training stage`.
- Fill in the template: why, what is changing, how to test.
- Ask at least one teammate for a review, and do not merge a pull request without an approval.
- Keep each pull request small and focused on one ticket.

## Checks before pushing

Run these checks locally before you push:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run dvc dag
uv run pytest
```

Use `uv run ruff format .` to fix formatting and `uv run ruff check --fix .` to fix the lint errors that can be fixed automatically.

## Data and credentials

- Data is versioned with DVC, never with Git: everything under `data/` is ignored. After `uv run dvc repro`, commit the updated `dvc.lock` and run `uv run dvc push`.
- Credentials never go into Git. Connect to DagsHub with a personal access token, not your password, stored in `.dvc/config.local`, a file that only exists on your machine:

  ```bash
  uv run dvc remote modify origin --local auth basic
  uv run dvc remote modify origin --local user <dagshub-username>
  uv run dvc remote modify origin --local password <dagshub-token>
  ```

- Other tokens (MLflow, Hugging Face) go in a local `.env` file, which is also ignored.
- Before committing, check `git status` and make sure none of these files appear.
- If a secret is committed by mistake and not pushed yet, remove it from the commit before pushing. If it was already pushed, change or revoke it immediately: the repository is public, and deleting the file in a later commit does not remove it from the history.
