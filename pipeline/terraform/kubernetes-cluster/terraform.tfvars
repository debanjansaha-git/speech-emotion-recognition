# GCP Provider Variables
project_id = "ie7374mlops"
region     = "us-east1"
zone       = "us-east1-a"

# VPC Variables
vpc_name = "vpc-main"

# Subnets Variables
subnets = [
  {
    subnet_name   = "subnet-01"
    subnet_ip     = "10.0.1.0/24"
    subnet_region = "us-east1-a"
  },
  {
    subnet_name   = "subnet-02"
    subnet_ip     = "10.0.2.0/24"
    subnet_region = "us-east1-b"
  }
]

# NAT Gateway Variables
nat_name                = "nat"
router_name             = "router"
nat_external_ip_count   = 1

# Kubernetes Cluster Variables
cluster_name            = "kube-primary-cluster"
cluster_version         = "latest"
machine_type            = "e2-medium"
disk_size_gb            = 30
node_count              = 3
node_locations          = ["us-east1-a", "us-east1-b"]
cluster_channel         = "REGULAR"

# Firewall Variables
firewall_rules = [
  {
    name        = "allow-internal"
    description = "Allow internal traffic"
    ranges      = ["10.0.0.0/8"]
    ports       = ["all"]
    protocol    = "all"
  },
  {
    name        = "allow-external"
    description = "Allow external SSH and HTTP traffic"
    ranges      = ["0.0.0.0/0"]
    ports       = ["22", "80", "443"]
    protocol    = "tcp"
  }
]

# Kubernetes Variables
kube_gen_node_count        = 1
kube_gen_node_machine_type = "e2-small"
spot_node_min_count        = 0
spot_node_max_count        = 10
spot_node_machine_type     = "e2-medium"

# Service Account Variables
service_account_name = "mlops-ser-sa"
roles = [
  "roles/container.clusterAdmin",
  "roles/container.developer"
]
