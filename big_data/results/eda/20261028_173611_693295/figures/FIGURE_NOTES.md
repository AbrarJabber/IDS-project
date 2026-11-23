# EDA figure notes

- **01_class_distribution.png**: Each percentage uses the total number of records in its own dataset as the denominator.

- **02_attack_families.png**: Normal traffic is excluded. Percentages use attack records as the denominator. Consult the CSV files for category counts.

- **03_protocol_service_train.png**: Groups are selected by record count. Group sizes are shown beside their names. Associations do not establish causation.

- **04_numeric_quantiles_train.png**: Approximate quantiles from Spark. The symlog scale includes zero. Features have different units; compare classes within each feature. Invalid, nonfinite and negative values were counted separately and excluded from these numerical summaries.

- **03_protocol_service_test.png**: Groups are selected by record count. Group sizes are shown beside their names. Associations do not establish causation.

- **04_numeric_quantiles_test.png**: Approximate quantiles from Spark. The symlog scale includes zero. Features have different units; compare classes within each feature. Invalid, nonfinite and negative values were counted separately and excluded from these numerical summaries.

- **03_protocol_service_full.png**: Groups are selected by record count. Group sizes are shown beside their names. Associations do not establish causation.

- **04_numeric_quantiles_full.png**: Approximate quantiles from Spark. The symlog scale includes zero. Features have different units; compare classes within each feature. Invalid, nonfinite and negative values were counted separately and excluded from these numerical summaries.

- **05_data_quality.png**: Null attack_cat values for normal full-data records can be structural. Dash placeholders are counted separately and are not automatically treated as errors. Check numeric_quality CSVs for numeric parsing and validity issues.

- **05_predictor_duplicates.png**: Extra duplicates equal total records minus unique predictor vectors. This is not the percentage of all rows in repeated groups. Train/test overlap is reported separately.