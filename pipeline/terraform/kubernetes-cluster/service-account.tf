# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/google_service_account
# Create the Google Cloud Service Account (GCSA)
resource "google_service_account" "service_account" {
  project    = var.project_id
  account_id = var.service_account_name
}

# Assign roles to the GCSA
# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/google_project_iam
resource "google_project_iam_member" "service_account_roles" {
  for_each = toset(var.roles)
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.service_account.email}"
}

# Create the Kubernetes Service Account (KSA) in the default namespace
resource "kubernetes_service_account" "ksa" {
  metadata {
    name      = var.service_account_name
    namespace = "default"
    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.service_account.email
    }
  }
  depends_on = [
    google_container_cluster.primary,
  ]
}

# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/google_service_account_iam
# Bind the GCSA to the KSA
resource "google_service_account_iam_binding" "workload_identity_binding" {
  service_account_id = google_service_account.service_account.name
  role               = "roles/iam.workloadIdentityUser"
  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[default/${var.service_account_name}]",
  ]
}