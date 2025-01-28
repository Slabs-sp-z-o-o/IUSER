# Main repository

Purpose of this repository is to run integration tests of our whole system.

## Deployment on GCP

### Setup
1. Think of a name of your project (e.g. `pwrability-stage`).
2. Go to pwrability-common and make duplicate .env_test with some new name (e.g. `.env_stage`).
3. Replace all text occurrences of old project_id (e.g. `pwrability-test-2`) 
   with new one (e.g. `pwrability-stage`) in new env file.
4. Add new ELIF statement to `scripts/gcp.sh::get_env_filename` to associate PROJECT_ID with environment file.
5. Make sure to update common submodule in all further repositories which will be mentioned in this instruction.
   ```
   git submodule update --remote --force common
   ```
6. Set $ENV_FILE variable to point to your new env file (e.g export ENV_FILE='.env_stage').
7. Load environment variables:
   ```
   source common/$ENV_FILE && export $(cut -d= -f1 common/$ENV_FILE)
   ```
8. Create project on GCP. You can do it with Web UI or through command line:
   ```
   gcloud projects create $PROJECT_ID
   ```
9. Associate your newly created project with a billing account. You can do it with Web UI.
10. Update gcloud config to use your newly created project
   ```
   gcloud config set project $PROJECT_ID
   ```
11. Go to pwrability-cloud repository
12. Create bucket for preserving Terraform state:
   ```
   ./infrastructure/create_remote_backend.sh
   ```
13. Create directory for new Terraform configuration file. For example:
   ```
   NEW_ENV_DIR=stage
   mkdir -p infrastructure/environments/$NEW_ENV_DIR
   ```
14. Copy arbitrary old Terraform configuration file (main.tf) 
   to newly created directory
15. Replace some values in newly created configuration file (main.tf):
    1. Terraform backend reference:
    
    Replace:
      ```
     backend "gcs" {
       bucket = "pwrability-test-3-terraform-state"
     }
      ```
      With:
      ```
     backend "gcs" {
       bucket = "<YOUR_PROJECT_ID>-terraform-state"
     }
      ```
    2. Environment file reference
        
    Replace:
      ```
     locals {
      env = sensitive({for tuple in regexall("(.*)=(.*)", file("../../../common/.env_test3")) : tuple[0] => tuple[1]})
      }
      ```
      With:
      ```
     locals {
      env = sensitive({for tuple in regexall("(.*)=(.*)", file("../../../common/<YOUR_NEW_ENV_FILE>")) : tuple[0] => tuple[1]})
      }
      ```
16. Go to `infrastructure/environments/$NEW_ENV_DIR` directory
17. Enable (do it through Web UI, https://console.cloud.google.com/apis/library)
    1. Cloud Resource Manager API
    2. Identity and Access Management (IAM) API
    3. Cloud SQL Admin API
    4. Cloud Pub/Sub API
    5. Compute Engine API 
    6. Cloud Build API
18. Authenticate with GCloud SDK to be able to use terraform commands for GCP projects
   ```
   gcloud auth application-default login
   ```    
19. Initialize terraform modules:
   ```
   terraform init   
   ```
20. Create resources:
   ```
   terraform apply   
   ```

21. Go to pwrability-nodes repository
22. Load environment variables:
   ```
   source common/$ENV_FILE && export $(cut -d= -f1 common/$ENV_FILE)
   ```
23. Initialize tables in Nodes database:
   ```
   ./flush_and_init_gcp_db.sh
   ```
24. Deploy Nodes API to Cloud Run:
   ```
   ./deploy_api_to_cloud_run.sh
   ```

25. Go to pwrability-ml repository
26. Load environment variables:
   ```
   source common/$ENV_FILE && export $(cut -d= -f1 common/$ENV_FILE)
   ```
27. Push ML code to container registry
   ```
   ./send_docker_to_registry.sh
   ```
28. Deploy code to VMs on GCP
   ```
   ./cloud_deployment/deploy_to_gcp.sh
   ```
29. If you want to run e2e tests, make sure to fill your test data bucket with required csv files.




## Utility

### Create common submodule directory (for the first time)

`git submodule update --init --remote --force common`

### Update common submodule directory

`git submodule update --remote --force common`

### Activate environment variables

#### pwrability-development
`source common/.env_development && export $(cut -d= -f1 common/.env_development)`

#### Integration tests:

First, remember to set GOOGLE_CREDENTIALS_PATH variable
to point to your JSON gcp service account key.

`export GOOGLE_CREDENTIALS_PATH=<path_to_your_file>`

Copy files from '<main-project>-test-data' to '<new-project>-test-data'

Then, run:
(Change env file name in 'run_tests_on_dev_in_docker.sh' and 'run_tests_on_dev_no_docker.sh')
`./run_tests_on_dev_in_docker.sh`

#### Notebooks:

`./run_notebooks.sh`

Click on a link provided in console output.
Notebooks are in `notebooks` directory.
You need to have both SA keys for dev and prod environments to use 
notebooks properly.