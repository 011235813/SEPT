# Single Episode Policy Transfer (SEPT)

This is the code for experiments in the paper [Single Episode Policy Transfer in Reinforcement Learning](https://arxiv.org/abs/1910.07719), published in ICLR 2020. Ablations and baselines are included.

## Prerequisites

- Python 3.6
- TensorFlow 1.10.1


## Navigation

* `alg/` - Implementation of SEPT, ablations and baselines.
  - `alg/configs/` - A collection of JSON config files, one for each method on each domain.
* `results/` - Results of training and testing will be stored in subfolders here. Each independent training run will create a subfolder that contains the final Tensorflow model, performance log file, and timing. All test runs for a method on a domain will be stored in a single aggregate subfolder. For example, 5 parallel independent training runs may produce `results/hiv_sept_1`,...,`results/hiv_sept_5`, and test results will be stored in `results/hiv_sept`.
* `hip-mdp-public` - Contains code for environments used in [Killian et al. 2017](https://github.com/dtak/hip-mdp-public). We provide new top-level scripts, one for training, and one for testing the saved models with the single test episode constraint.

## Domains

Choice of experimental domain is selected by `config['main']['domain']` within the JSON config files.

* `2D` - source located in `hip-mdp-public/grid_simulator/grid.py`
* `acrobot` - source located in `hip-mdp-public/acrobot_simulator/acrobot_py3.py`
* `hiv` - source located in `hip-mdp-public/hiv_simulator/hiv.py`

## Example

### Training SEPT on 2D navigation

* Check general settings in `alg/configs/config_2d_sept.json`. E.g.
  * `"domain" : "2D"`
  * `"N_seeds" : 20"`
  * `"dir_name" : "2D_sept"`
- `cd` into the `alg` folder
- Execute training script: `python train_multiprocess.py configs/config_2d_sept.json`
- Periodic logging and final model are stored in `results/2D_sept_<int>`, where `int` ranges from `dir_idx_start` to `N_seeds` (see the configs).

### Testing SEPT on 2D navigation

* Keep the same settings in `alg/configs/config_2d_sept.json` as those used for training
* `cd` into the `alg` folder. 
* Execute test script `python test_multiprocess.py configs/config_2d_sept.json`
* Results will be stored in `test_<int>.csv` and `test_time_<int>.pkl` in `results/2D_sept/`

## License

SEPT is distributed under the terms of the BSD-3 license. All new contributions must be made under this license.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

SPDX-License-Identifier: BSD-3

LLNL-CODE-805017