data "terraform_remote_state" "gcp_environment" {
  backend = "gcs"
  config = {
    bucket = "tf-deploy-kube-test"
    prefix = "terraform/state/gcp-environment"
  }
}


