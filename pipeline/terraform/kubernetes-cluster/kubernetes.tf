# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/container_cluster
resource "google_container_cluster" "primary" {
  name                     = var.cluster_name
  location                 = var.region
  project                  = var.project_id
  remove_default_node_pool = true
  initial_node_count       = 1
  network                  = data.terraform_remote_state.gcp_environment.outputs.vpc_self_link
  subnetwork               = data.terraform_remote_state.gcp_environment.outputs.subnets_self_links["subnet-01"]
  # network                  = google_compute_network.main.self_link
  # subnetwork               = google_compute_subnetwork.subnets[0].self_link
  logging_service          = "logging.googleapis.com/kubernetes"
  monitoring_service       = "monitoring.googleapis.com/kubernetes"
  networking_mode          = "VPC_NATIVE"
  node_locations           = var.node_locations

  

  release_channel {
    channel = var.cluster_channel
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = "k8s-pod-range-0"
    services_secondary_range_name = "k8s-service-range-0"
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  #   Jenkins use case
  #   master_authorized_networks_config {
  #     cidr_blocks {
  #       cidr_block   = "10.0.0.0/18"
  #       display_name = "private-subnet-w-jenkins"
  #     }
  #   }

  node_config {
  machine_type = var.kube_od_node_machine_type
  oauth_scopes = [
    "https://www.googleapis.com/auth/logging.write",
    "https://www.googleapis.com/auth/monitoring",
  ]

  metadata = {
    "disable-legacy-endpoints" = "true"
  }

  workload_metadata_config {
    mode = "GKE_METADATA"
  }

  labels = { # Update: Replace with desired labels
    "environment" = "test"
    "team"        = "devops"
  }
}
  addons_config {
    http_load_balancing {
      disabled = true
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
  }
}

output "cluster_endpoint" {
  value = google_container_cluster.primary.endpoint
}

output "cluster_ca_certificate" {
  value = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
}

resource "null_resource" "authenticate_with_gcloud" {
  depends_on = [data.terraform_remote_state.gcp_environment]

  provisioner "local-exec" {
    command = <<EOT
    echo '${data.terraform_remote_state.gcp_environment.outputs.gcsa_key}' > /tmp/gcsa_key.json
    gcloud auth activate-service-account --key-file=/tmp/gcsa_key.json
    EOT
    interpreter = ["/bin/bash", "-c"]
  }
}


# Setup kubeconfig after cluster creation
resource "null_resource" "kubeconfig_setup" {
  depends_on = [
    google_container_cluster.primary,
    null_resource.authenticate_with_gcloud
  ]

  provisioner "local-exec" {
    command = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${var.region} --project ${var.project_id}"
    environment = {
      KUBECONFIG = "${path.module}/kubeconfig"
    }
  }
}

# provider "kubernetes" {
#   config_path = "${path.module}/kubeconfig"
# }
