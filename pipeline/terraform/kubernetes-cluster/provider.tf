# Configure the Google Cloud Platform (GCP) provider.
# https://registry.terraform.io/providers/hashicorp/google/latest/docs
provider "google" {
  project = var.project_id                # Sets the GCP project ID to manage resources
  region  = var.region                    # Sets the default region for resources
}

provider "kubernetes" {
  host = "https://${google_container_cluster.primary.endpoint}"
  token = data.google_client_config.provider.access_token
  cluster_ca_certificate = base64decode(
    google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  )
}
# Terraform configuration block.
# https://www.terraform.io/language/settings/backends/gcs
terraform {
  # Configures the backend for storing Terraform state files in Google Cloud Storage (GCS).
  backend "gcs" {
    bucket = "tf-deploy-kube-test"                  # Specifies the name of the GCS bucket to store Terraform state files.
    prefix = "terraform/state/kubernetes-setup"     # Sets the path prefix within the bucket where state files are stored, allowing for organized state management.
  }
  # Defines required providers and their versions.
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}