# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/google_service_account
resource "google_service_account" "service_account" {
  project    = var.project_id
  account_id = var.service_account_name
}

# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/google_project_iam
resource "google_project_iam_member" "service_account_roles" {
  for_each = toset(var.roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.service_account.email}"
}

# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/google_service_account_iam
resource "google_service_account_iam_member" "workload_identity_user" {
  service_account_id = google_service_account.service_account.id
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[staging/${var.service_account_name}]"
}