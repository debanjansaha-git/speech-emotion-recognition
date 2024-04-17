# TerraForm by Hashicorp

Why use Terraform?

Terraform is cloud-agnostic and allows writing Infrastructure deployments as a Code (IaaC), meaning, you can connect terraform to any cloud providers, and automatically allocate resources by simply executing the terraform modules, without the hassle of going through the manual resouce allocation process.
Also, Terraform supports descriptive coding which allows ease in interpretation of the resource allocation and following industry best-practices such as the principle of "Least Priviledge".

In order to execute resource allocation using code (IaaC), a pre-requisite is to install Terraform.

For Mac users, please execute the commands:

## Installing HomeBrew

HomeBrew is the unofficial missing package manager for Mac OS. In order to install HomeBrew use the following command:

```bash
export HOMEBREW_NO_INSTALL_FROM_API=1
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```

## Installing Terraform

For Mac users, Terraform can be easily installed with HomeBrew. Please execute the commands:

```bash
brew tap hashicorp/tap
```

```bash
brew install hashicorp/tap/terraform
```

```bash
brew update
```

```bash
brew upgrade hashicorp/tap/terraform
```

Verify the installation by:

```bash
terraform -help
```

For users on all other platforms, please follow the guidelines given [HERE](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)

