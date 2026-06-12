# Train models

### Input

To train a model, users need to provide:

- a demographic model in [DEMES](https://popsim-consortium.github.io/demes-spec-docs/main/introduction.html) format
- a `GAISHI` [configuration file](https://xin-huang.github.io/gaishi/userguide/configuration/) in YAML format

### Feature-vector models

Feature-vector training is used for:

- `logistic_regression`
- `extra_trees_classifier`

Run training with:

    gaishi train \
        --demes examples/demog/ArchIE_3D19.yaml \
        --config examples/configs/lr.config.yaml \
        --output examples/results/models/example.lr.onnx

To train an extra-trees classifier, use the same command with `model.name` set to `extra_trees_classifier` in the configuration file.

### UNet++ models

UNet++ training uses genotype matrices stored in HDF5 format.

Run training with:

    gaishi train \
        --demes examples/demog/ArchIE_3D19.yaml \
        --config examples/configs/unet.config.yaml \
        --output examples/results/models/example.unet.safetensors

### Only simulate training data

Users can run simulation without training a model:

    gaishi train \
        --demes examples/demog/ArchIE_3D19.yaml \
        --config examples/configs/lr.config.yaml \
        --only-simulation

### Output

For feature-vector models, `gaishi train` creates:

- a training table: `<output_dir>/<output_prefix>.tsv`
- a trained model file in the [ONNX](https://onnx.ai/) format specified by `--output`

For UNet++ models, `gaishi train` creates:

- a training HDF5 file: `<output_dir>/<output_prefix>.h5`
- a trained model checkpoint in the [Safetensors](https://huggingface.co/docs/safetensors/index) format specified by `--output`

### Settings

| Argument | Description |
| - | - |
| `--demes` | Path to the DEMES demographic model file. |
| `--config` | Path to the `GAISHI` configuration YAML file. |
| `--output` | Path to the trained model output file. Required unless `--only-simulation` is used. |
| `--only-simulation` | Run simulation only and skip model training. |
