# Contributing to Heart Risk AI

First off, thank you for considering contributing to Heart Risk AI. It's people like you that make the open source community such a great community!

## Where do I go from here?

If you've noticed a bug or have a feature request, [make one](https://github.com/JuanInfante122/heart-risk-model.git/issues/new)! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

## Fork & create a branch

If this is something you think you can fix, then [fork Heart Risk AI](https://github.com/JuanInfante122/heart-risk-model.git/fork) and create a branch with a descriptive name.

A good branch name would be (where issue #33 is the ticket you're working on):

```bash
git checkout -b 33-add-new-feature
```

## Get the test suite running

Make sure you're running the tests before you start making changes. We've got a `Makefile` that makes it easy to set up the project and run the tests.

```bash
make install
```

## Model Training and Development

If you are contributing to model improvements, please follow these guidelines:

1.  **Create a New Script:** Do not modify existing training scripts. Create a new version (e.g., `train_model_v3.py`) to ensure reproducibility of previous results.
2.  **Document Your Process:** Use the `AdvancedHeartRiskPredictor` class as a template. Ensure your script clearly outlines the steps: data loading, feature engineering, model selection, and evaluation.
3.  **Automated Results:** The training script automatically saves results, plots, and the final model to a timestamped directory (e.g., `training_results_YYYYMMDD_HHMMSS/`). This is the preferred way to track experiments.
4.  **Update Documentation:** If your new model is chosen as the primary one, update the `readme.md` and any other relevant documentation with the new results and methodology.

## Implement your fix or feature

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first :smile_cat:

## Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with Heart Risk AI's master branch:

```bash
git remote add upstream git@github.com:yourusername/heart-risk-ai.git
git checkout master
git pull upstream master
```

Then update your feature branch from your local copy of master, and push it!

```bash
git checkout 33-add-new-feature
git rebase master
git push --force-with-lease origin 33-add-new-feature
```

Finally, go to GitHub and [make a Pull Request](https://github.com/JuanInfante122/heart-risk-model.git/compare) :D

## Keeping your Pull Request updated

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code has changed, and that you need to update your branch so it's easier to merge.

To learn more about rebasing and merging, check out this guide on [merging vs. rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing).

## Thank you!

Thank you for contributing!