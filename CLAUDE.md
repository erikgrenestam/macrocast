# Macroeconomic forecasting tool

### python env
Use uv to manage the python venv.

### example data
Example data for training and testing can be found in @data/example_data.parquet. Note that the data is in long format. Use the value_date column to infer the frequency of each variable (identified by the column `internal_series_name`). For duplicates within `value_date` by `internal_series_name`, use the `value` associated with the latest `valid_to`.

### pymc-bart
API-reference: https://www.pymc.io/projects/bart/en/latest/api_reference.html
Examples: https://www.pymc.io/projects/bart/en/latest/examples.html

### SQL queries
SQL queries are to be executed using the package dnsql and dnsql.execute_query(query_string, cfg='datahub'). This is a restricted package and is not included in the venv. Treat the function as working.

### Legacy code
The folder legacy contains legacy code. This is code can be considered obsolete.