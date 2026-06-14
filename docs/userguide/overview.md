# Overview

`GAISHI` uses simulated training data and machine learning models to detect introgressed sites or haplotypes.

There are two subcommands in `GAISHI`:

- `train`
- `infer`

To display help information for each subcommand, users can use:

    gaishi subcommand -h

For example:

    gaishi train -h

**Note:** In this manual, we define the population without introgressed fragments as the **reference population**, the population receiving introgressed fragments as the **target population**, and the population donating introgressed fragments as the **source population**.

The example commands assume that users have cloned the [GAISHI](https://github.com/xin-huang/gaishi) GitHub repository and run the commands from the root directory of the repository:

```
git clone https://github.com/xin-huang/gaishi
cd gaishi
```
