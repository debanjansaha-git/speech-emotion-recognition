# Dynamically create Google Cloud Firewall rules based on provided specifications.
# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_firewall
resource "google_compute_firewall" "dynamic_firewall_rules" {
  # Iterates over the 'firewall_rules' variable, transforming it into a map where the key is the rule name. 
  # This allows for unique identification and management of each rule.
  for_each = {for fr in var.firewall_rules : fr.name => fr}

  name    = each.value.name                     # Sets the name for each firewall rule
  network = google_compute_network.main.name    # Associates the firewall rule with a specific networ

  # Configures the allowed traffic for the rule, specifying the protocol and ports based on the input variable.
  allow {
    protocol = each.value.protocol
    ports    = each.value.ports
  }

  # Sets the source IP ranges that the rule applies to, allowing traffic from these ranges to the specified ports and protocol.
  source_ranges = each.value.ranges
}