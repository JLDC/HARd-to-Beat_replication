# Replication package for "HARd-to-Beat: The Overlooked Impact of Rolling Windows in the Era of Machine Learning"

Jonathan Chassot and Francesco Audrino

## Overview & Contents

#### `wrds`

```
wrds/
├── aggregate_data.py   # Code to aggregate the downloaded TAQ data
├── query_taq.py        # Code to download the TAQ data from WRDS
└── utils.py            # Diverse helper functions
```

##### 1. Downloading the TAQ data

`wrds/query_taq.py` can be run to download the TAQ data for a given stock and a given date range.

For instance, to download the TAQ data for Tesla from January 2020 to February 2020, run
```
python wrds/query_taq.py 
    --dir "path/to/output/dir" # The output directory, store 1 pickle file per day
    --daterange "20200101-20200201" # Date range to download MMDDYYYY-MMDDYYYY
    --credentials "path/to/credentials/file" # Path to the credentials file
    --symbol "TSLA" # Symbol root and suffix, separated by a dot, e.g., GOOG.A
```

The `credentials` file should contain the credentials for the WRDS database, i.e., the username on the first line and the password on the second line.

##### 2. Aggregating the TAQ data

`wrds/aggregate_data.py` can be run to aggregate the downloaded TAQ data.

```
python wrds/aggregate_data.py
    --input "path/to/input/dir"     # The input directory containing the TAQ data
    --output "path/to/output/dir"   # The output directory for the aggregated data
    --no-overwrite                  # Do not overwrite existing files
    --processes N                   # Number of processes to use
```


