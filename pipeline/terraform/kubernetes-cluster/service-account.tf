# Create the Kubernetes Service Account (KSA) in the default namespace
resource "kubernetes_service_account" "ksa" {
  metadata {
    name      = var.kube_service_account_name
    namespace = "default"
    annotations = {
      "iam.gke.io/gcp-service-account" = data.terraform_remote_state.gcp_environment.outputs.service_account_email
    }
  }
  automount_service_account_token = true
  depends_on = [
    null_resource.authenticate_with_gcloud,
    google_container_cluster.primary,
    google_container_node_pool.spot_pool,
    google_container_node_pool.on_demand_pool
  ]
}

# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/google_service_account_iam
# Bind the GCSA to the KSA
resource "google_service_account_iam_binding" "workload_identity_binding" {
  # service_account_id = google_service_account.service_account.name
  service_account_id = "projects/${var.project_id}/serviceAccounts/${data.terraform_remote_state.gcp_environment.outputs.service_account_email}"
  role               = "roles/iam.workloadIdentityUser"
  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[default/${var.kube_service_account_name}]",
  ]
}