# Configure the Google Cloud Platform (GCP) provider.
# https://registry.terraform.io/providers/hashicorp/google/latest/docs
provider "google" {
  project = var.project_id                # Sets the GCP project ID to manage resources
  region  = var.region                    # Sets the default region for resources
}

# Terraform configuration block.
# https://www.terraform.io/language/settings/backends/gcs
terraform {
  # Configures the backend for storing Terraform state files in Google Cloud Storage (GCS).
  backend "gcs" {
    bucket = "tf-deploy-kube-test"                # Specifies the name of the GCS bucket to store Terraform state files.
    prefix = "terraform/state/gcp-environment"    # Sets the path prefix within the bucket where state files are stored, allowing for organized state management.
  }
  # Defines required providers and their versions.
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}