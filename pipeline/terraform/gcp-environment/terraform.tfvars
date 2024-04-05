# GCP Provider Variables
project_id = "ie7374mlops"                              # The ID of the GCP project where all resources will be created.
region     = "us-east1"                                 # region where the resources will be created
zone       = "us-east1-a"                               # availability zone

# VPC Variables
vpc_name = "vpc-main"                                   # The name of the Virtual Private Cloud network, a private network within GCP.

# Subnets Variables
subnets = [                                             # Defines details for creating subnetworks within the VPC
  {
    subnet_name   = "subnet-01"                         # Name of the subnet-01
    subnet_ip     = "10.10.0.0/24"                      # IP CIDR range of the subnet-01
    subnet_region = "us-east1"                          # Region of subnet-01
    subnet_zones  = ["us-east1-a", "us-east1-b"]        # Availability Zones for subnet-01
  },
  {
    subnet_name   = "subnet-02"                         # Name of the subnet-02
    subnet_ip     = "10.10.1.0/24"                      # IP CIDR range of the subnet-02
    subnet_region = "us-east1"                          # Region of subnet-02
    subnet_zones  = ["us-east1-c", "us-east1-d"]        # Availability Zones for subnet-02
  }
]

# NAT Gateway Variables
nat_name                = "nat"                         # NAT gateway name
router_name             = "router"                      # IGW router name
nat_external_ip_count   = 1                             # number of NAT IP addresses (external IPs)

# Kubernetes Cluster Variables
cluster_name            = "kube-primary-cluster"        # K8s cluster name
cluster_version         = "latest"                      # K8s cluster version
machine_type            = "e2-medium"                   # K8s node machine type
disk_size_gb            = 30                            # K8s node persistent storage
node_count              = 3                             # K8s number of nodes
node_locations          = ["us-east1-b", "us-east1-c"]  # K8s node placement availability zones
cluster_channel         = "REGULAR"                     # K8s cluster channel

# Firewall Variables
firewall_rules = [                                      # Firewall Rules
  {
    name        = "allow-internal"                      # internal communication b/w GCP resources
    description = "Allow internal traffic"              # description
    ranges      = ["10.0.0.0/8"]                        # IP CIDR range
    ports       = []                                    # use all ports
    protocol    = "all"                                 # use all protocols
  },
  {
    name        = "allow-external"                      # allow internet access
    description = "Allow external SSH and HTTP traffic" # description
    ranges      = ["0.0.0.0/0"]                         # internet
    ports       = ["22", "80", "443"]                   # ssh, http, https ports
    protocol    = "tcp"                                 # protocol
  }
]

# Kubernetes Variables
kube_gen_node_count        = 1                          # K8s standard cluster node count
kube_gen_node_machine_type = "e2-small"                 # K8s standard cluster node machine type
spot_node_min_count        = 0                          # K8s min node count using spot instance 
spot_node_max_count        = 10                         # K8s max node count using spot instance 
spot_node_machine_type     = "e2-medium"                # K8s spot instance machine type

# Service Account Variables
service_account_name = "sa-terraform"                   # GCSA account name
roles = [
  "roles/container.clusterAdmin",                       # manage Kubernetes cluster in GKE
  "roles/container.developer",                          # cluster application deployment
  "roles/compute.networkAdmin"                          # manage VPC resources in GCP
]

