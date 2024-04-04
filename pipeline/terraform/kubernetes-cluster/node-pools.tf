# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/google_service_account
# resource "google_service_account" "k8s-sa" {
#   account_id = "k8s-sa"
# }

# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/container_node_pool
# On-Demand Node Pool
resource "google_container_node_pool" "on_demand_pool" {
  name       = "on-demand-node-pool"
  location   = var.zone
  project    = var.project_id
  cluster    = google_container_cluster.primary.id
  node_count = var.kube_od_node_count
  autoscaling {
    min_node_count = var.kube_od_node_min_count
    max_node_count = var.kube_od_node_max_count
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    labels = {
      env = var.project_id
    }
    preemptible  = false
    machine_type = var.kube_od_node_machine_type
    tags         = ["gke-on-demand-node"]
    metadata = {
      disable-legacy-endpoints = "true"
    }
    # service_account = data.terraform_remote_state.gcp_environment.outputs.service_account_email
    taint = [{
      effect = "PREFER_NO_SCHEDULE"
      key    = "type"
      value  = "on-demand"
    }]
  }
}

# Spot Instance Pool
resource "google_container_node_pool" "spot_pool" {
  provider = google-beta
  name    = "spot-node-pool"
  location   = var.zone
  project    = var.project_id
  cluster = google_container_cluster.primary.id
  node_count = 1
  autoscaling {
    min_node_count = var.spot_node_min_count
    max_node_count = var.spot_node_max_count
  }
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    labels = {
      env = var.project_id
    }
    spot = true
    machine_type = var.spot_node_machine_type
    tags         = ["gke-spot-node"]
    metadata = {
      disable-legacy-endpoints = "true"
    }

    # service_account = data.terraform_remote_state.gcp_environment.outputs.service_account_email
  }
}