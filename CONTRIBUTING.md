# Contributing to AmpyFin

Thank you for your interest in contributing! This guide reviews how to be a part of AmpyFin.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to keep our community respectful and welcoming.

This applies to everyone collaborating with the codebase, [Issue Tracker](https://github.com/AmpyFin/ampyfin/issues), and [Discussion Board](https://github.com/AmpyFin/ampyfin/discussions).

## Reporting Issues

Our [GitHub Issue Tracker](https://github.com/AmpyFin/ampyfin/issues) is the place to start. If you've encountered a problem in AmpyFin, check to see it hasn't already been reported.

Each issue needs to include a title and a clear description of the problem. Add as much information as you can to make it easy to reproduce the issue and determine a fix.

Once the issue is open, be patient. Others can find your issue, and confirm it, or may collaborate with you on a fix. If you know how to fix the issue, open a pull request.

## Help Resolve Existing Issues

Beyond reporting issues, you can help resolve existing ones by providing feedback.

### Verify a report

If you can reproduce a reported issue, add a comment confirming it. Provide any additional information to reproduce. Anything you can do to make reports easier to reproduce helps! Whether you end up writing the code yourself or not.

### Test a patch

You can also help by looking at [pull requests](https://github.com/AmpyFin/ampyfin/pulls) that have been submitted. To test someone's changes, create a dedicated branch:
```bash
git checkout -b test_branch
```

Then, use their remote branch to update your codebase. Example, SomeUser has a fork and pushed to branch 'feature':
```bash
git remote add SomeUser https://github.com/SomeUser/ampyfin.git
git pull SomeUser feature
```

*Alternatively, use the [GitHub CLI tool](https://cli.github.com) to checkout their pull request.*

## Contributing to the Documentation

## Contributing to the Code

We recommend using the provided Dev Container for development. This ensures a consistent environment with all necessary tools pre-installed.

### Development Environment

#### Using GitHub Codespaces

With codespaces enabled, you can fork and use codespaces on GitHub. The Codespace will be initialized with all required dependencies and allow you to run tests.

#### Using VS Code Remote Containers

With [Visual Studio Code](https://code.visualstudio.com) and [Docker](https://www.docker.com/) installed, you can use the [VS Code remote containers plugin](https://code.visualstudio.com/docs/remote/containers-tutorial). The plugin will read the `.devcontainer` configuration in the repository and build the Docker container locally.

#### Using Dev Container CLI

With [npm](https://github.com/npm/cli) and [Docker](https://www.docker.com/) installed, you can run [Dev Container CLI](https://github.com/devcontainers/cli) to utilize the `.devcontainer` configuration from the command line.

```bash
npm install -g @devcontainers/cli
cd AmpyFin
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash
```

#### Local Development

See the [Installation](README#installation) section for how to set up local development. This is the harder approach because installing dependencies may be OS specific.

### Clone the Repository

Clone the AmpyFin repository:
```bash
git clone https://github.com/AmpyFin/ampyfin.git
```
Create a topic branch:
```bash
cd ampyfin
git checkout -b new_branch
```

*This branch will only exist on your local computer and personal GitHub repository, not the AmpyFin repository*

### Pip install

Install required dependencies:
```bash
pip install -r requirements.txt
```

### Make Changes & Follow Coding Conventions

This project uses pre-commit hooks to ensure code quality. Before pushing your changes, please run pre-commit locally:

Install pre-commit:
```bash
pip install pre-commit
```

Install the Git hooks:
```bash
pre-commit install
```

Run pre-commit on all files:
```bash
pre-commit run --all-files
```

Fix any issues identified by the pre-commit hooks before pushing your changes. Running pre-commit locally will make the PR process smoother and reduce the need for revision requests.

### Commit Changes

Commit the changes to Git:
```bash
git commit -a
```

A well-formatted and descriptive commit message is helpful to others for understanding why the change was made. Take time to write one. A good commit message will have the following:
```bash
Short summary (ideally 50 characters or less)

More detailed description, if necessary. Each line should wrap at
72 characters. Be descriptive - even if you think the commit
content is obvious, it may not be obvious to others. Add any
description that is already present in the relevant issues; it should
not be necessary to visit a webpage to check history.

The description section can have multiple paragraphs.

You can also add bullet points:

- make a bullet point by starting a line with either a dash (-)
  or asterisk (*)

- wrap lines at 72 characters, and indent any additional lines
  with 2 spaces for readability
```

*Please squash your commits into a single commit when appropriate. This simplifies future cherry picks and keeps the git log clean.*

### Update Your Branch

It's very likely other changes to main have happened while you worked. To get new changes in main:
```bash
git checkout main
git pull --rebase
```

Reapply your patch on top of latest changes:
```bash
git checkout new_branch
git rebase main
```

Resolve any conflicts, then push rebase to GitHub:
```bash
git push --force-with-lease
```

The AmpyFin repository base does not allow force pushing, but you are able to force push your fork. When rebasing, this is a requirement since the history has changed.

### Fork

Go to [ampyfin](https://github.com/AmpyFin/ampyfin) and press "Fork". Add the new remote to your local repository:
```bash
git remote add fork https://github.com/your-user-name/ampyfin.git
```

If you cloned your repository directly from AmpyFin/ampyfin, you already have the upstream remote. If not, add it:
```bash
git remote add AmpyFin https://github.com/AmpyFin/ampyfin.git
```

Get new commits and branches from the official repository:
```bash
git fetch AmpyFin
```

Merge new content:
```bash
git checkout main
git rebase AmpyFin/main
git checkout new_branch
git rebase AmpyFin/main
```

Update fork:
```bash
git push fork main
git push fork new_branch
```

### Opening a Pull Request

Go to your ampyfin repo (https://github.com/your-user-name/ampyfin) and click "Pull Requests". Your pull should target the base repository `AmpyFin/ampyfin` and the branch `main`. The head repo will be your changes (`your-user-name/ampyfin`), with the branch name you gave. Verify changeset, fill in details, and create the pull request.

### Getting Feedback

After submitting your pull request, here's what to expect:

- **Review Process**: Most PRs go through several iterations before merging. Contributors may have different opinions, and revisions are common.

- **Response Time**: Please be patient with feedback. AmpyFin contributors have varying availability. Initial responses may take a few days.

- **Engagement Tips**: While waiting, consider reviewing other open PRs. Respond promptly to feedback when received. Avoid directly pinging individual maintainers.

- **Approval Process**: Only designated maintainers can merge code changes. An "approval" from a non-maintainer is valuable feedback but doesn't guarantee merging.

- **Iterative Improvement**: Be open to constructive criticism and willing to make changes. The goal is high-quality contributions that align with project standards.

Remember that the collaborative review process leads to better code and helps maintain AmpyFin's quality standards.

#### *Squashing Commits*

We generally prefer pull requests that contain a single, well-formed commit. Squashing combines multiple commits into one, which helps in several ways:

- Makes it easier to understand the change as a whole
- Simplifies reverting changes if needed
- Keeps the git history clean and meaningful
- Makes backporting to other branches more straightforward

If you have multiple commits in your PR, we might ask you to squash them before merging. Here's how to do it:

**Interactive rebase to the number of commits you want to squash:**
```bash
git rebase -i HEAD~3 # squash the last 3 commits
```

**In the editor that opens, change "pick" to "squash" (or "s") for all commits except the first one:**
```bash
pick abc1234 First commit message
squash def5678 Second commit message
squash ghi9101 Third commit message
```

**Save and close the file. Another editor will open to edit the combined commit message.**

**After squashing, you'll need to force push to your branch:**
```bash
git push fork new_branch --force-with-lease
```

Refresh the pull request on GitHub and see that it has been updated. Remember that squashing rewrites history, so only do this on branches that only you are working on or after coordinating with collaborators.

#### *Updating an Existing Pull Request*

After creating a pull request, you might receive feedback requesting changes to your code. Rather than opening a new PR, you should update your existing one by modifying your branch and pushing the updates.

If you've already pushed your branch and need to modify existing commits, a regular push will fail because the branch histories no longer match. In this case, you'll need to force push to your branch, as described earlier in the squashing commits section:

```bash
git commit --amend
git push fork new_branch --force-with-lease
```

**Important safety note:** Always use `--force-with-lease` instead of `-f` or `--force`. This option checks that you're not overwriting others' work that you haven't seen yet, providing a safeguard against accidentally removing commits pushed by collaborators.

## Contributors

Thank you for your contributions!
