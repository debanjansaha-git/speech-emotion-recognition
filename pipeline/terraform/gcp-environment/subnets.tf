# Dynamically creates subnetworks within a specified VPC in Google Cloud Platform.
# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_subnetwork
resource "google_compute_subnetwork" "subnets" {
  count                    = length(var.subnets)                        # Uses the count meta-argument to create multiple subnetworks based on the length of the 'subnets' variable.
  name                     = var.subnets[count.index].subnet_name
  ip_cidr_range            = var.subnets[count.index].subnet_ip         # Sets the IP CIDR range for each subnet
  region                   = var.subnets[count.index].subnet_region     # Assigns each subnetwork to a specific region
  network                  = google_compute_network.main.id             # Links each subnetwork to the previously defined VPC network by referencing its ID.
  private_ip_google_access = true                                       # Enables private access to Google services without assigning external IP addresses to VM instances

  # Defines a secondary IP range for Kubernetes pod IPs in each subnetwork, using a calculated CIDR range to avoid overlap.
  secondary_ip_range {
    range_name    = "k8s-pod-range-${count.index}"
    ip_cidr_range = "10.42.${count.index * 4}.0/22"
  }
  
  # Defines a second secondary IP range for Kubernetes service IPs in each subnetwork, using another calculated CIDR range.
  secondary_ip_range {
    range_name    = "k8s-service-range-${count.index}"
    ip_cidr_range = "10.142.${count.index * 4}.0/22"
  }
}

output "subnets_self_links" {
  value = {for s in google_compute_subnetwork.subnets : s.name => s.self_link}
}

output "subnets_secondary_ranges" {
  value = {for s in google_compute_subnetwork.subnets : s.name => s.secondary_ip_range}
}
