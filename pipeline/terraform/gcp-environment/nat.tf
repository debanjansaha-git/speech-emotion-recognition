# Reserve external IP addresses for NAT.
# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_address
resource "google_compute_address" "nat_ips" {
  count        = var.nat_external_ip_count            # The number of external IPs for NAT
  name         = "${var.nat_name}-${count.index}"     
  region       = var.region                           # Specifies the region for the external IP addresses
  address_type = "EXTERNAL"         # Sets the address type to external, as these IPs are used for outbound internet access.
  network_tier = "PREMIUM"          # Chooses the "PREMIUM" network tier for the IP addresses, offering higher performance and reliability.
}

# Create a NAT gateway on a Google Cloud Router.
# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_router_nat
resource "google_compute_router_nat" "nat" {
  name   = var.nat_name
  router = google_compute_router.router.name                    # Associates the NAT gateway with a specific router
  region = var.region                                           # Specifies the region for the NAT gateway
  nat_ip_allocate_option             = "MANUAL_ONLY"            # Configures the NAT gateway to use manually specified external IPs for NAT, rather than auto-allocating them.
  source_subnetwork_ip_ranges_to_nat = "LIST_OF_SUBNETWORKS"    # Sets up NAT for all IP ranges of the subnetworks, enabling outbound internet access for all internal IPs within those subnetworks.

  # Dynamically configures subnetworks for NAT, allowing each to use the NAT gateway for internet access.
  dynamic "subnetwork" {
    for_each = google_compute_subnetwork.subnets
    content { 
      name                    = subnetwork.value.self_link      # References the self_link of each subnetwork, ensuring the NAT gateway can correctly identify and route traffic from them.
      source_ip_ranges_to_nat = ["ALL_IP_RANGES"]
    }
  }

  # Specifies the external IP addresses reserved for the NAT gateway, ensuring outbound traffic can be correctly mapped and routed.
  nat_ips = [for ip in google_compute_address.nat_ips : ip.self_link]
}
