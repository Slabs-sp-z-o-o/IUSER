# AIGO API end-to-end tests

## How to run?

1. Go to repo base directory.
2. Make sure SSH tunnel to GCP instance is activated.

   `sh ./ml/tests/utils/tunel_google.sh`

   Note: You may need to also append configuration to your `~/.ssh/config`
         file in order to have this script work:

   ```text
   host 34.78.121.80
     user <your_configured_user_with_ssh_access_to_ml_api_gcp_vm>
   ```

3. Export your Google credentials:

   `export GOOGLE_APPLICATION_CREDENTIALS=pathToYourGooglePrivateKey.json`

4. Run: `./run_aigo_tests.sh`

## Structure

- **temporary_data/** - place where temporal files from Storage will be saved
                        during tests setup, if loading data to BigQuery
                        will be necessary
- **tests_results/** - place where plots of predictions will be saved
- **base.py** - base class for E2E tests
- **nodes_setup.py** - parameters for nodes and meters created for tests
- **scenarios_handlers.py** - all helper classes for a scenario related operations
- **conftest.py** - inputs for tests, like possible algorithms, possible horizons
- **scenarios.py** - concrete scenarios for tests. It contains information
  such as:
  - `endogenous` for scenario
  - `nodes_setup_scenario` to setup node with meters
  - `telemetry_data_scenario` to use for given test
  - `weather_forecasts_data_scenario` to use for given test
  - `config` of meters in a node (should reflect SQL state to simple things up
    without need to communicate with SQL database to create training dataframe
  - `create_and_train_request_body` is self-explaining
  - `predict_request_body` is self-explaining

- **telemetry_data_scenarios.py** - configuration to provide reproducibility
  of data for tests. One data scenario can be used for multiple scenarios.
  It contains information such as:
  - `project_id`
  - `dataset`
  - `table`
  - `columns` to use when downloading scenario_data from BigQuery
  - `columns_to_change` from base csv file in Storage
  - `meter_ids` - mapping for `columns_to_change`

    IMPORTANT! Even if you are not renaming any columns you need to provide
               them, so you could do something like:
               `{'meter_1001' : 'meter_1001'}` if you don't change anything.

  - `gateway_id` - mapping for `columns_to_change` (IMPORTANT! Same rule as with
    `meter_ids` applies)
  - `source_csv` and `source_bucket` - place where source CSV of given
    scenario data lies
  - `validation_data` - used by function which checks if given scenario data
    is already loaded to BQ

- **weather_forecasts_data_scenarios.py** - analogical to telemetry data
  scenarios (`location` - location of a node for given scenario_data)

- **test_end_to_end.py** - end-to-end tests of ML API

## Usage of scenarios

### Applying selected scenario to the test

Add:

```python
@pytest.mark.parametrize('scenario_name', ['your_scenario_name'])
```

To use selected scenario from your `scenarios.py` file.

### Logic

After adding scenario to the test, automatic data setup in BigQuery will be
performed for both telemetry and weather forecasts.

#### Telemetry

1. Data presence is checked by performing simple query using `validation_data`
   from scenario_data. If number of records matches, all checks are stopped -
   we have (MOST LIKELY) our data, if not - we will need to load data to BQ.
2. Presence of source CSV file is checked in a local directory.
   If file is not present, we download it from Storage.
3. Dataframe is created from source CSV. Column renaming is performed
   to match given SQL setup.
4. Transformed Dataframe is loaded into BigQuery.

#### Weather forecasts

Here, mechanism is more straight-forward, since number of weather forecasts is
so small that we can simply flush everything related to given scenario_data
before each test.

1. Weather forecasts data for given scenario is deleted from BQ.
2. Source CSV is downloaded from Storage.
3. Dataframe is created from source CSV.
4. Dataframe is loaded into BQ.
