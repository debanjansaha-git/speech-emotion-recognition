variable "project_id" {
  description = "The GCP project ID."
  type        = string
}

variable "region" {
  description = "The region where resources will be created."
  type        = string
}

variable "zone" {
  description = "The zone within the region where resources will be created."
  type        = string
}

variable "vpc_name" {
  description = "The name of the VPC to be created."
  type        = string
  default     = "vpc-main"
}

variable "subnets" {
  description = "A list of subnets to be created within the VPC."
  type        = list(object({
    subnet_name   = string
    subnet_ip     = string
    subnet_region = string
    subnet_zones  = list(string)
  }))
}

variable "nat_name" {
  description = "The name of the NAT gateway."
  type        = string
  default     = "nat"
}

variable "router_name" {
  description = "The name of the router."
  type        = string
  default     = "router"
}

variable "nat_external_ip_count" {
  description = "The number of external IPs for the NAT gateway."
  type        = number
  default     = 1
}

variable "cluster_name" {
  description = "The name of the Kubernetes cluster."
  type        = string
  default     = "kube_primary-cluster"
}

variable "cluster_version" {
  description = "The version of the Kubernetes cluster."
  type        = string
  default     = "latest"
}

variable "machine_type" {
  description = "The machine type for the Kubernetes nodes."
  type        = string
  default     = "e2-medium"
}

variable "disk_size_gb" {
  description = "The disk size in GB for the Kubernetes nodes."
  type        = number
  default     = 30
}

variable "node_count" {
  description = "The number of nodes in the Kubernetes cluster."
  type        = number
  default     = 3
}

variable "node_locations" {
  description = "Additional zones where nodes should be created in a multi-zonal cluster."
  type        = list(string)
  default     = ["us-east1-b", "us-east1-c"]
}

variable "cluster_channel" {
  description = "The release channel of the Kubernetes cluster."
  type        = string
  default     = "REGULAR"
}

variable "firewall_rules" {
  description = "A list of firewall rules to be created."
  type        = list(object({
    name        = string
    description = string
    ranges      = list(string)
    ports       = list(string)
    protocol    = string
  }))
}

variable "kube_od_node_count" {
  description = "Number of nodes in the on-demand node pool"
  type        = number
  default     = 1
}

variable "kube_od_node_machine_type" {
  description = "Machine type for the on-demand node pool"
  type        = string
  default     = "e2-medium"
}

variable "kube_od_node_min_count" {
  description = "Minimum number of nodes in the spot node pool"
  type        = number
  default     = 0
}

variable "kube_od_node_max_count" {
  description = "Maximum number of nodes in the spot node pool"
  type        = number
  default     = 10
}

variable "spot_node_min_count" {
  description = "Minimum number of nodes in the spot node pool"
  type        = number
  default     = 0
}

variable "spot_node_max_count" {
  description = "Maximum number of nodes in the spot node pool"
  type        = number
  default     = 10
}

variable "spot_node_machine_type" {
  description = "Machine type for the spot node pool"
  type        = string
  default     = "e2-medium"
}

variable "goog_service_account_name" {
  description = "The name of the google cloud service account."
  type        = string
  default     = "mlops-gcsa"
}

variable "kube_service_account_name" {
  description = "The name of the Kubernetes service account."
  type        = string
  default     = "mlops-ksa"
}

variable "roles" {
  description = "A list of roles to be assigned to the service account."
  type        = list(string)
}
