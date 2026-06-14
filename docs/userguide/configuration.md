# Configuration

`GAISHI` uses one [YAML](https://en.wikipedia.org/wiki/YAML) configuration file for both training and inference.

The configuration file has three top-level blocks:

- `simulation`
- `preprocessing`
- `model`

The `train` command uses `simulation` and `model`.

The `infer` command uses `preprocessing` and `model`.

### Feature-vector configuration

Feature-vector configuration is used for `logistic_regression` and `extra_trees_classifier`.

    simulation:
      sim_type: "feature_vector"
      nrep: 100
      nref: 5
      ntgt: 5
      ref_id: "Reference"
      tgt_id: "Target"
      src_id: "Source"
      ploidy: 2
      is_phased: true
      seq_len: 50000
      mut_rate: 1.2e-8
      rec_rate: 1e-8
      nprocess: 2
      feature_config_file: "examples/configs/ArchIE.features.yaml"
      intro_prop: 0.7
      non_intro_prop: 0.3
      output_prefix: "example.lr"
      output_dir: "examples/results/data/training"
      seed: 12345
      nfeature: 100
      is_shuffled: false
      force_balanced: false
      keep_sim_data: false

    preprocessing:
      process_type: "feature_vector"
      vcf_file: "examples/data/example.vcf"
      chr_name: "1"
      ref_ind_file: "examples/data/ref.ind.list"
      tgt_ind_file: "examples/data/tgt.ind.list"
      win_len: 50000
      win_step: 50000
      is_phased: true
      feature_config_file: "examples/configs/ArchIE.features.yaml"
      output_dir: "examples/results/data/infer"
      output_prefix: "example.lr"
      nprocess: 2
      ploidy: 2

    model:
      name: "logistic_regression"
      params:
        solver: "newton-cg"
        max_iter: 10000
        random_state: 12345

### Genotype-matrix configuration

Genotype-matrix configuration is used for `unet++`.

    simulation:
      sim_type: "genotype_matrix"
      nrep: 100
      nref: 50
      ntgt: 50
      ref_id: "Reference"
      tgt_id: "Target"
      src_id: "Source"
      ploidy: 2
      is_phased: true
      seq_len: 50000
      mut_rate: 1.2e-8
      rec_rate: 1e-8
      nprocess: 2
      output_prefix: "example.unet"
      output_dir: "examples/results/data/training"
      seed: 12345
      force_balanced: false
      keep_sim_data: false
      num_polymorphisms: 192
      num_genotype_matrices: 100

    preprocessing:
      process_type: "genotype_matrix"
      vcf_file: "examples/data/example.vcf"
      chr_name: "1"
      ref_ind_file: "examples/data/ref.ind.list"
      tgt_ind_file: "examples/data/tgt.ind.list"
      anc_allele_file: null
      num_polymorphisms: 192
      step_size: 56
      output_dir: "examples/results/data/infer"
      output_prefix: "example.unet"
      ploidy: 2
      is_phased: true
      nprocess: 2

    model:
      name: "unet++"
      params:
        add_rnn: false
        learning_rate: 0.001
        batch_size: 32
        n_early: 10
        n_epochs: 20
        min_delta: 1e-4
        val_prop: 0.05
        seed: 12345

### `simulation`

| Field | Description |
| - | - |
| `sim_type` | Simulation type. Use `feature_vector` or `genotype_matrix`. |
| `nrep` | Number of simulation replicates per batch. |
| `nref` | Number of reference individuals. |
| `ntgt` | Number of target individuals. |
| `ref_id` | Reference population ID in the DEMES model. |
| `tgt_id` | Target population ID in the DEMES model. |
| `src_id` | Source population ID in the DEMES model. |
| `ploidy` | Sample ploidy. |
| `is_phased` | Whether the simulated data are phased. |
| `seq_len` | Simulated sequence length in base pairs. |
| `mut_rate` | Mutation rate per base pair per generation. |
| `rec_rate` | Recombination rate per base pair per generation. |
| `nprocess` | Number of processes. Default: `1`. |
| `output_prefix` | Prefix of output files. |
| `output_dir` | Output directory. |
| `seed` | Random seed. |
| `force_balanced` | Enforce balanced introgressed and non-introgressed classes. Default: `false`. |
| `keep_sim_data` | Keep raw simulation files. Default: `false`. |

Feature-vector only:

| Field | Description |
| - | - |
| `feature_config_file` | Path to the feature configuration file. |
| `intro_prop` | Minimum introgressed proportion for labeling a window as introgressed. |
| `non_intro_prop` | Maximum introgressed proportion for labeling a window as non-introgressed. |
| `nfeature` | Number of feature vectors to generate. |
| `is_shuffled` | Shuffle feature rows before output. Default: `true`. |

Genotype-matrix only:

| Field | Description |
| - | - |
| `num_polymorphisms` | Number of polymorphic sites per genotype matrix. |
| `num_genotype_matrices` | Number of genotype matrices to generate. |

### `preprocessing`

| Field | Description |
| - | - |
| `process_type` | Preprocessing type. Use `feature_vector` or `genotype_matrix`. |
| `vcf_file` | Path to the input VCF/BCF file. |
| `chr_name` | Chromosome name to process. |
| `ref_ind_file` | File containing reference individual IDs. |
| `tgt_ind_file` | File containing target individual IDs. |
| `output_dir` | Output directory. |
| `output_prefix` | Prefix of output files. |
| `nprocess` | Number of processes. Default: `1`. |
| `ploidy` | Sample ploidy. Default: `2`. |
| `is_phased` | Whether the VCF genotypes are phased. Default: `true`. |
| `anc_allele_file` | Optional ancestral allele file. Default: `null`. |

Feature-vector only:

| Field | Description |
| - | - |
| `win_len` | Window length in base pairs. |
| `win_step` | Window step size in base pairs. |
| `feature_config_file` | Path to the feature configuration file. |

Genotype-matrix only:

| Field | Description |
| - | - |
| `num_polymorphisms` | Number of polymorphic sites per genotype matrix. |
| `step_size` | Step size between genotype matrices. |

### `model`

| Field | Description |
| - | - |
| `name` | Model name. Use `logistic_regression`, `extra_trees_classifier`, or `unet++`. |
| `params` | Model-specific parameters. |

For `logistic_regression`, parameters are passed to [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

For `extra_trees_classifier`, parameters are passed to [sklearn.ensemble.ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).

For `unet++`, commonly used parameters include:

| Field | Description |
| - | - |
| `add_rnn` | Use the RNN/4-channel UNet++ variant. Default: `false`. |
| `batch_size` | Batch size. Default: `32`. |
| `n_early` | Early stopping patience. Default: `10`. |
| `n_epochs` | Maximum number of epochs. Default: `100`. |
| `learning_rate` | Learning rate. Default: `0.001`. |
| `min_delta` | Minimum validation-loss improvement. Default: `1e-4`. |
| `val_prop` | Validation proportion. Default: `0.05`. |
| `seed` | Random seed. Default: `null`. |
| `device` | Device, such as `cpu` or `cuda:0`. Default: `null`. |
| `num_workers` | Number of PyTorch DataLoader workers. Default: `0`. |
| `use_amp` | Use CUDA AMP. Default: `false`. |
| `recent_window` | Window size for recent training-log metrics. Default: `500`. |
