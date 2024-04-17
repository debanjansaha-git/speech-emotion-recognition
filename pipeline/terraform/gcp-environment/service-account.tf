# Create the Google Cloud Service Account (GCSA)
# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/google_service_account
resource "google_service_account" "service_account" {
  project      = var.project_id               # The project ID
  account_id   = var.service_account_name     # The unique ID to assign to the service account
  display_name = "GCP Service Account for Kubernetes"
}

# Assign roles to the GCSA
# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/google_project_iam
resource "google_project_iam_member" "service_account_roles" {
  for_each = toset(var.roles)     # Iterates over the list of roles
  project = var.project_id        # The project ID
  role    = each.value            # The IAM role to assign
  member  = "serviceAccount:${google_service_account.service_account.email}"
}


# Create a service account key for the GCSA
resource "google_service_account_key" "gcsa_key" {
  service_account_id = google_service_account.service_account.name
}

# Output the service account email
output "service_account_email" {
  value = google_service_account.service_account.email
  sensitive = true
}

# Output the path of the created service account key file
output "gcsa_key" {
  value = base64decode(google_service_account_key.gcsa_key.private_key)
  sensitive = true
}

# Output the service account key details
output "gcsa_key_details" {
  value = google_service_account_key.gcsa_key
  sensitive = true
}