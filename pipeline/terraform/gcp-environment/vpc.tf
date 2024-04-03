# Define a resource for enabling the Google Compute Engine API on a project.
# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/google_project_service
resource "google_project_service" "compute" {
  service = "compute.googleapis.com"
  disable_on_destroy   = false
}

# Define a resource for enabling the Google Kubernetes Engine (GKE) API on a project.
resource "google_project_service" "container" {
  service = "container.googleapis.com"
  disable_on_destroy   = false
}

# Create a Google Cloud VPC (Virtual Private Cloud) network.
# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_network
resource "google_compute_network" "main" {
  name                            = var.vpc_name
  routing_mode                    = "REGIONAL"        # The network routing mode, which determines how network traffic is routed.
  auto_create_subnetworks         = false             # Disables automatic creation of subnetworks, giving more control over subnetwork configuration.
  mtu                             = 1460              # Sets the Maximum Transmission Unit for the VPC. The default for Google Cloud is 1460 bytes.
  delete_default_routes_on_create = false             # If set to true, default routes to the internet are deleted when the network is created. It's false here, so default routes remain.
  
  # specifies the pre loading of dependecies
  depends_on = [
    google_project_service.compute,
    google_project_service.container
  ]
}

output "vpc_self_link" {
  value = google_compute_network.main.self_link
}
