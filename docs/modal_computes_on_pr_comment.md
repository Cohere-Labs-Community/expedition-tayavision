# Running Modal Scripts from PR Comments

Run Modal scripts directly from pull request comments using the `/run` command. The CI/CD workflow validates the command, executes the script on Modal, and reports status back to the PR.

## Usage

Comment on any open pull request:

```
/run scripts/modal_<name>.py
```

### Available Scripts

| Script | Description |
|--------|-------------|
| `scripts/modal_pytest.py` | Run tests |
| `scripts/modal_train_alignment.py` | Train vision-language alignment |
| `scripts/modal_train_instruct.py` | Train instruction tuning |
| `scripts/modal_eval.py` | Run evaluation |
| `scripts/modal_eval_aligned_tokens.py` | Evaluate aligned tokens |
| `scripts/modal_eval_mlp_l2_norm.py` | Evaluate MLP L2 norm |
| `scripts/modal_download.py` | Download model/data assets |

### Examples

```
/run scripts/modal_pytest.py
/run scripts/modal_train_alignment.py
```

## How It Works

1. You comment `/run scripts/modal_<name>.py` on a PR
2. The workflow adds a 🚀 reaction to your comment
3. A status comment is posted: ⏳ **Running** ...
4. The PR branch is checked out and the script is validated
5. The Modal script is executed via `modal run -m <module>`
6. The status comment is updated to ✅ **Done** or ❌ **Failed** (with a link to logs)

## Requirements

### Repository Secrets

The following secrets must be configured in **Settings → Secrets and variables → Actions**:

- `MODAL_TOKEN_ID` — Modal API token ID
- `MODAL_TOKEN_SECRET` — Modal API token secret

### Actions Permissions

Under **Settings → Actions → General**:

- **Workflow permissions** must be set to **Read and write permissions** (needed to post comments and reactions on PRs)

### Important Note

The workflow file (`.github/workflows/ci-cd.yml`) must exist on the **default branch** (`main`) for `issue_comment` triggers to work. Changes to the workflow on a feature branch won't take effect until merged.

## Security

- Only scripts matching the pattern `scripts/modal_<word_chars>.py` are accepted
- The script path is validated by regex before execution, preventing command injection
- The workflow verifies the script file exists in the repo before running
