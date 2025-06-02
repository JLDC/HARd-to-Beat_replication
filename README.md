# Replication package for "HARd-to-Beat: The Overlooked Impact of Rolling Windows in the Era of Machine Learning"

Jonathan Chassot and Francesco Audrino

## Overview & Contents

```bash
.
├── data/            # Data loading and preprocessing
├── losses/          # Loss functions
├── models/          # Models
├── utils/           # Diverse helper functions
├── wrds/            # Downloading and aggregating the TAQ data from WRDS
├── config.yaml      # Configuration file for run_models.py
├── config_eval.yaml # Configuration file for run_eval.py
├── environment.yaml # Python environment file for the project
├── run_eval.py      # Code to evaluate the forecasts
└── run_models.py    # Code to run the models
```

### 1. Preparing the data

The `wrds` directory contains the code to download and aggregate the data from [Wharton Research Data Services (WRDS)](https://wrds-www.wharton.upenn.edu/). Due to the data being proprietary, we cannot share the data and only provide the code to download it given the sufficient authorization.

```bash
wrds/
├── aggregate_data.py   # Code to aggregate the downloaded TAQ data
├── query_taq.py        # Code to download the TAQ data from WRDS
└── utils.py            # Diverse helper functions
```

##### 1.1. Downloading the TAQ data

`wrds/query_taq.py` can be run to download the TAQ data for a given stock and a given date range.

For instance, to download the TAQ data for Tesla from January 2020 to February 2020, run
```bash
python wrds/query_taq.py 
    --dir "path/to/output/dir"                  # The output directory, store 1 pickle file per day
    --daterange "20200101-20200201"             # Date range to download MMDDYYYY-MMDDYYYY
    --credentials "path/to/credentials/file"    # Path to the credentials file
    --symbol "TSLA"                             # Symbol root and suffix, separated by a dot, e.g., GOOG.A
```

The `credentials` file should contain the credentials for the WRDS database, i.e., the username on the first line and the password on the second line.

##### 1.2. Aggregating the TAQ data

`wrds/aggregate_data.py` can be run to aggregate the downloaded TAQ data.

```bash
python wrds/aggregate_data.py
    --input "path/to/input/dir"     # The input directory containing the TAQ data
    --output "path/to/output/dir"   # The output directory for the aggregated data
    --no-overwrite                  # Do not overwrite existing files
    --processes int                 # Number of processes to use
```


### 2. Running the models

The `run_models.py` script can be run to train all models from the paper and produce the forecasts as follows:

```bash
python run_models.py
    --config "path/to/config.yaml" # Path to the configuration file (yaml)
    --verbose                       # Whether to print verbose output
    --overwrite                     # Whether to overwrite existing results
    --warnings                      # Whether to filter convergence warnings
```

The `config.yaml` file contains the configuration for the models, attributes and values can be changed to run different models, use different data, use different RV kernels, change the training, validation and test windows.

```yaml
data:
  path: "path/to/data" # Path to the directory containing the aggregated TAQ data
  start_date: "YYYY-MM-DD" # Start date for the training window
  start_valid: "YYYY-MM-DD" # Start date for the validation window
  start_test: "YYYY-MM-DD" # Start date for the test window
  kernel: None # Kernel to use for the RV, can be "preavg" or None
  index: [all, dow30, nasdaq100] # Indices to run the models on

models:
  har:
    training_window: int # Training window in days
    reestimation_frequency: int # Reestimation frequency in days

  lasso:
    hyperparams:
      alpha: # Log-spaced grid of regularization parameter
        from: float # Lower bound of the grid
        to: float # Upper bound of the grid
        total: int # Number of values in the grid

  ffnn:
    hyperparams:
      ffnn__hidden_layer_sizes: list[list[int]] # Network architectures
      ffnn__alpha: list[float] # Regularization parameters

  gbt:
    hyperparams:
      max_iter: list[int] # Number of trees
      learning_rate: list[float] # Learning rates
      max_depth: list[int] # Maximum depth of the trees

  rf:
    hyperparams:
      n_estimators: list[int] # Number of trees
      min_samples_leaf: list[int] # Minimum number of samples per leaf
      max_features: list[float|string] # Maximum number of features
```

### 3. Evaluating the forecasts and producing the tables / figures

The `run_eval.py` script can be run to evaluate the forecasts and produce the result tables / figures of the paper as follows:

```bash
python run_eval.py
    --config "path/to/config_eval.yaml" # Path to the configuration file (yaml)
```

The `config_eval.yaml` file contains the configuration for the evaluation, attributes and values can be changed to specify which tables / figures to produce and how to format the model names.

```yaml
kernel: list[str] # RV kernels to use, values can be "rv05" or "rv05_preavg"
index: list[str] # Indices to evaluate the models on, values can be "all", "dow30" or "nasdaq100"
models:
  model_name: "Model name" # Model name to use in the tables / figures
  model_name_pooled: "Model name (pooled)" # Model name to use in the tables / figures (pooled)
  model_name_novix: "Model name (no VIX)" # Model name to use in the tables / figures (no VIX)
  model_name_pooled_novix: "Model name (pooled, no VIX)" # Model name to use in the tables / figures (pooled, no VIX)
```

