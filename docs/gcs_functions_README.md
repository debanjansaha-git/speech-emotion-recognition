# Google Cloud Storage Utility Functions

The gcs_functions.py contains utility functions to facilitate file uploads to Google Cloud Storage (GCS) buckets. Below is a description of each function and its purpose:

## `upload_file(bucket_name, source_file_name, destination_file_name)`

Uploads a single file to a specified location in a Google Cloud Storage bucket.

- Parameters:
  - `bucket_name` (str): The name of the Google Cloud Storage bucket.
  - `source_file_name` (str): The local path of the file to be uploaded.
  - `destination_file_name` (str): The name of the file as it will appear in the bucket.

## `upload_directory(bucket_name, source_directory, workers=8)`

Uploads every file in a directory, including all files in subdirectories, to a Google Cloud Storage bucket.

- Parameters:
  - `bucket_name` (str): The name of the Google Cloud Storage bucket.
  - `source_directory` (str): The directory path on the local machine containing files to be uploaded.
  - `workers` (int, optional): The maximum number of processes to use for the operation. Default is 8.

## `upload_file_to_folder(destination_blob_name, path_to_file, bucket_name, destination_path="")`

Uploads a single file to a specified folder within a Google Cloud Storage bucket.

- Parameters:
  - `destination_blob_name` (str): The name of the file as it will appear in the bucket.
  - `path_to_file` (str): The local path of the file to be uploaded.
  - `bucket_name` (str): The name of the Google Cloud Storage bucket.
  - `destination_path` (str, optional): The folder path within the bucket where the file will be uploaded. Default is an empty string (root of the bucket).

## `upload_directory_to_folder(source_directory, gcs_destination)`

Uploads files from a local directory to a specified folder within a Google Cloud Storage bucket.

- Parameters:
  - `source_directory` (str): The local directory path containing the files to be uploaded.
  - `gcs_destination` (str): The destination path in GCS where the files will be uploaded.

