# Run inference

### Input

To run inference, users need to provide:

- a [trained model](https://xin-huang.github.io/gaishi/userguide/train/#output)
- a `GAISHI` [configuration file](https://xin-huang.github.io/gaishi/userguide/configuration/) in YAML format

### Feature-vector models

Feature-vector inference is used for models trained with:

- `logistic_regression`
- `extra_trees_classifier`

Run inference with:

    gaishi infer \
        --model examples/results/models/example.lr.joblib \
        --config examples/configs/lr.config.yaml \
        --output examples/results/infer/example.lr.pred.tsv

### UNet++ models

UNet++ inference uses genotype matrices stored in HDF5 format.

Run inference with:

    gaishi infer \
        --model examples/results/models/example.unet.safetensors \
        --config examples/configs/unet.config.yaml \
        --output examples/results/infer/example.unet.pred.tsv

### Output

For feature-vector models, the output table contains one row per window and sample:

| Column | Description |
| - | - |
| `Chromosome` | Chromosome name. |
| `Start` | Window start position. |
| `End` | Window end position. |
| `Sample` | Target sample or haplotype name. |
| `Non_Intro_Prob` | Probability of the non-introgressed class. |
| `Intro_Prob` | Probability of the introgressed class. |

For UNet++ models, the output table contains one row per site and sample:

| Column | Description |
| - | - |
| `Chromosome` | Chromosome name. |
| `Position` | Site position. |
| `Sample` | Target sample or haplotype name. |
| `Non_Intro_Prob` | Probability of the non-introgressed class. |
| `Intro_Prob` | Probability of the introgressed class. |

### Settings

| Argument | Description |
| - | - |
| `--model` | Path to the trained model file. |
| `--config` | Path to the `GAISHI` configuration YAML file. |
| `--output` | Path to the inference output file. |
