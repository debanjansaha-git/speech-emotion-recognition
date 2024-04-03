# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_subnetwork
resource "google_compute_subnetwork" "subnets" {
  count                    = length(var.subnets)
  name                     = var.subnets[count.index].subnet_name
  ip_cidr_range            = var.subnets[count.index].subnet_ip
  region                   = var.subnets[count.index].subnet_region
  network                  = google_compute_network.main.id
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "k8s-pod-range-${count.index}"
    ip_cidr_range = "10.142.${count.index * 4}.0/22"
  }
  secondary_ip_range {
    range_name    = "k8s-service-range-${count.index}"
    ip_cidr_range = "10.142.${count.index * 4 + 64}.0/22"
  }
}