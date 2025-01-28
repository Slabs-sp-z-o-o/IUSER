## 🧩 Requirements

* 🐋 Docker
* ⚪ gcloud CLI

## 📜 Instruction for running predictions

1. `git clone git@bitbucket.org:enalpha/pwrability-main.git`
2. `cd pwrability-main`
3. `git submodule update --init --remote --force common`
4. `source common/.env_development && export $(cut -d= -f1 common/.env_development)`
5. Now, download you service account key JSON file from GCP
   console (https://cloud.google.com/iam/docs/keys-create-delete#creating)
6. `export GOOGLE_CREDENTIALS_PATH=<absolute_path_to_json_service_account>`
7. `gcloud init`
   * Login to your GCP account 
   * Select your project (pwrability-development)
8. `./run_tests_on_dev_in_docker.sh /app/tests/end_to_ends/test_end_to_end_hardcoded.py`
9. Press Enter to run the test
10. Check out the results in `tests/end_to_ends/tests_results` directory

## ⭐ Tips and tricks

* Feel free to modify parameters in SUMMER_MODEL in `tests/end_to_ends/tests_inputs/train_request_bodies.py`
    * One thing worth doing is to comment out whole `exogenous` property of `training_data`.
      This way, we are ignoring weather measurements and only use historical data. Since the model is designed to
      predict
      energy production, such change should hugely damage the model's performance.
* The test which is supposed to run is written in `tests/end_to_ends/test_end_to_end_hardcoded.py::TestEndToEndHardcoded::test_training_demoday`,
you can also change some things in there, like for example metric for which the evaluation will be shown.