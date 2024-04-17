# Setup Airflow on GCP

There are multiple ways of setting up Airflow on GCP. Airflow comes pre-installed with Cloud Composer, so that would be probably the easiest way to go. However, here we will manually deploy Airflow as a Docker containerized application with our custom settings adjusted for our needs.

Since, we are using a custom docker composer for Airflow, we need to host our Airflow server on a Google Cloud VM (GCE) and manually maintain this instance.

Please follow the step-by-step guidelines below to install and configure Airflow on GCP
1. Go to GCP marketplace, and find the custom image for Docker compose which comes pre-built with Docker, Docker compose, and other dependencies related to Docker. Or you can use this [Link](https://console.cloud.google.com/marketplace/product/cloud-infrastructure-services/docker-compose-ubuntu20)
2. Click on the 'Enable' button following which enable all the APIs required for this.
3. Once, in the page to provision the server, enter the details as:
   - Deployment Name: airflow-webserver
   - Zone: us-east-1b
   - Machine Type: General Purpose
   - Series: C3
   - Machine Type: c3-standard-22 (22 vCPU, 11 core, 88GB memory)
   - Boot Disk: Balanced Persistent Disk
   - Boot disk size in GB: 30 GB
 - Keep everything else as default and hit the "Deploy" button and wait for your VM to get deployed
4. From the Hamburger menu on the left-top corner, click on:
   - Compute Engine -> VM instances
   - Look at the bottom middle section for "Set up firewall rules"
   - Click on the button "CREATE FIREWALL RULE" in the top-middle section
   - Provide the following details:
     - Name: allow-airflow
     - Description: Allow Airflow port
     - Keep everything as default and scroll down
     - In the target tags enter "allow-airflow"
     - Source IPv4 range: 0.0.0.0/0
     - Scroll to the Protocols and ports, click on TCP and enter: 8080
     - Click on "CREATE"
     - A new rule with name "allow-airflow" will be created with Protocols / ports as "tcp: 8080", Action: allow
5. From the Hamburger menu on the left-top corner, click on: Compute Engine -> VM instances -> Click on the Name: airflow-webserver
6. Click on the "EDIT" button at the top:
7. Scroll to the bottom where it says "Firewalls" and check the boxes next to:
   - Allow HTTP traffic
   - Allow HTTPS traffic
8. In the "Network tags", enter:
   -  allow-airflow
-  Keep everything else as default as hit "SAVE"
9. Go to the VM instances (back->back) and find your VM instance named "airflow-webserver"
10. Click on the "Connect -> SSH" and allow "Authorize"
11. Finally, you should now be able to SSH into your VM instance
12. Inside the VM command shell execute the following commands:
    - sudo systemctl start docker
    - sudo service docker start
    - sudo docker run hello-world  # this should download the docker hello-world template and run in your VM just to check if Docker is running fine in your VM
    - now, run the following command to check if git is installed in your VM
      - git --version  ## this should show the current version of git installed in your VM
    - now, clone the GitHub repository in your VM as:
    - git clone https://github.com/debanjansaha-git/speech-emotion-recognition.git
    - This will clone the github repository in the folder called "speech-emotion-recognition"
13. Allow permissions for user to use docker
    - type "cd" to go to the root directory and run the following commands:
    - sudo groupadd docker
    - sudo usermod -aG docker ${USER}
    - logout  ## logout of VM or close the SSH terminal
14. Go to the VM instance page, and select the checkbox next to "airflow-webserver" and click on the "STOP" button, and wait for the instance to be stopped completely.
15. Select the webserver, and click on "START" button to reboot the server.
16. SSH into the server.
17. Navigate to the directory:
    - cd speech-emotion-recognition/pipeline/airflow/
18. Run the docker-compose file as:
    - docker compose up
    - This will pull all the docker images into your local instance, and start executing all the custom commands mentioned in the Dockerfile and the docker-compose.yaml file
    - Once everything is up and running and you see: "127.0.0.1 - - [21/Mar/2024:06:52:09 +0000] "GET /health HTTP/1.1" 200" messages getting populated, your Airflow webserver is up and running.
19. Go to the VM instances page, and copy the "External IP", for example: 34.23.208.131
20. Open a new tab, and paste the IP you copied in the previous step : 8080. For example, in this case it would be "34.23.208.131:8080" without the quotes
21. This will load up the airflow home page and ask you to login
22. Enter both username and password as "airflow"
23. You will be able to see the DAG pre-loaded as "Data Pipeline"
24. Run the DAG and monitor every step of the DAG
25. This DAG also requires uploading processed files from each step into a GCS bucket, for which you will need to place your GCS Bucket access key and rename it as (gcs_key.json) in the pipeline/airflow/config folder. Otherwise your data will not get uploaded into the GCS bucket.

Congratulations!! Now you have a working Airflow webserver running on GCP and performing your custom data pipeline workflows.

Once you are done with executing your workflow, don't forget to clean up the resources. 
 - You should always stop your VM instance to reduce the charges incurred. The VMs have persistent storage, meaning your data will remain in the VM even after you have stopped your VM
 - When you want to execute your VM again, "START" the VM from the VM instances page, and SSH into the VM. Follow the directions from Step 18
 - Since you have pre-allocated storage assigned to your VM, even in stopped state, you will be billed for your VM. Optionally, if you want to completely stop incurring any charges, you should delete your VM, which will give you a pop-up called: "Are you sure you want to delete instance "airflow-webserver-vm"? (This will also delete boot disk "airflow-webserver-vm")". You should select "Yes" to delete everything to completely stop incurring charges.
 - Additionally, you can also go ahead and delete the Firewall Rule created in Step 4, but that is optional. You will not be charged for having this rule not-deleted.

If you face any issues reach out to Debanjan for assistance.
Good Luck!!