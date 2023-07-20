## Contributing to PyTorch-OOD

All kinds of contributions are welcome, including but not limited to the following.

- Adding additional detectors 
- Fixing typos or bugs
- Improving documentation

### Workflow

1. fork and pull the latest PyTorch-OOD repository
2. checkout a new branch (do not use master/dev branch for PRs)
3. commit your changes
4. create a PR


If you plan to add some new features that involve large changes, it is encouraged to open an issue for discussion first.


### Code style

We use [pre-commit hook](https://pre-commit.com/) that checks and formats the code for automatically on every commit.
The config for a pre-commit hook is stored in [.pre-commit-config](./.pre-commit-config.yaml).

After you clone the repository, you will need to install pre-commit and initialize the pre-commit hook.

```shell
pip install -U pre-commit
```

From the repository folder

```shell
pre-commit install
```
