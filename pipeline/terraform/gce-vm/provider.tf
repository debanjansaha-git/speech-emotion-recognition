terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "5.21.0"
    }
  }
}

provider "google" {
  project = "ie7374mlops"
  region = "us-east1"
  zone = "us-east1-b"
  credentials = "../airflow/config/gcp_key_deb.json"
}