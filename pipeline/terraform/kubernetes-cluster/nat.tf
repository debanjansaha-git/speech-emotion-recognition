# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_router_nat
resource "google_compute_address" "nat_ips" {
  count        = var.nat_external_ip_count
  name         = "${var.nat_name}-${count.index}"
  region       = var.region
  address_type = "EXTERNAL"
  network_tier = "PREMIUM"
}

resource "google_compute_router_nat" "nat" {
  name   = var.nat_name
  router = google_compute_router.router.name
  region = var.region
  nat_ip_allocate_option             = "MANUAL_ONLY"
  source_subnetwork_ip_ranges_to_nat = "LIST_OF_SUBNETWORKS"

  dynamic "subnetwork" {
    for_each = google_compute_subnetwork.subnets
    content { 
      name                    = subnetwork.value.self_link
      source_ip_ranges_to_nat = ["ALL_IP_RANGES"]
    }
  }

  nat_ips = [for ip in google_compute_address.nat_ips : ip.self_link]
}
