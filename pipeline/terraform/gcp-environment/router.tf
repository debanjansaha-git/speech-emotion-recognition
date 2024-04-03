# Create a Google Cloud Router.
# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_router
resource "google_compute_router" "router" {
  name    = var.router_name
  region  = var.region                          # The region where the router will be located
  network = google_compute_network.main.id      # Associates the router with an existing VPC network by referencing the network's ID. 
  # This setup ensures the router can manage traffic within the specified network.
}