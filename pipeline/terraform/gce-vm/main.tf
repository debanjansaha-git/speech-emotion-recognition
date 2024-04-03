resource "google_compute_instance" "airflow_webserver" {
  name         = "airflow-webserver"
  zone         = "us-east1-b"
  machine_type = "n1-standard-16"
  tags         = ["allow-airflow", "http-server", "https-server"]

  boot_disk {
    initialize_params {
      image = "projects/cloud-infra-services-public/global/images/docker-compose-ubuntu-20-04-v10032023"
      size  = 30
    }
  }

  network_interface {
    network = "https://www.googleapis.com/compute/v1/projects/ie7374mlops/global/networks/default"
    access_config {}
  }

  metadata_startup_script = <<EOF
    #!/bin/bash
    exec > /tmp/startup-log.txt 2>&1
    sudo systemctl start docker
    apt-get update && apt-get install -y git
EOF

}

resource "google_compute_firewall" "allow_airflow" {
  name    = "allow-airflow"
  network = "https://www.googleapis.com/compute/v1/projects/ie7374mlops/global/networks/default"
  target_tags = ["allow-airflow"]

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  source_ranges = ["0.0.0.0/0"]
}
