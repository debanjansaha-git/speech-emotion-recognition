# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_firewall
resource "google_compute_firewall" "dynamic_firewall_rules" {
  for_each = {for fr in var.firewall_rules : fr.name => fr}

  name    = each.value.name
  network = google_compute_network.main.name

  allow {
    protocol = each.value.protocol
    ports    = each.value.ports
  }

  source_ranges = each.value.ranges
}