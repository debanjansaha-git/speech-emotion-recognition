resource "google_project_service" "project" {
  project            = var.project_id
  service            = "container.googleapis.com"
  disable_on_destroy = false
}

# Retrieve an access token as the Terraform runners
data "google_client_config" "provider" {
  depends_on = [
    google_container_cluster.primary
  ]
}