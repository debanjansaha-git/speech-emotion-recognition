"""
@Author: Gopi
"""
from google.cloud import storage
from pathlib import Path
from google.cloud.storage import Client, transfer_manager
import logging
import os


def authenticate(path_to_key):
    # set key credentials file path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_key
    return True

def upload_directory(bucket_name, source_directory, target_directory='', path_to_key='', max_workers=8):
    """Upload every file in a directory, including all files in subdirectories.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket to upload files to.
        source_directory (str): The directory path on the local machine containing files to be uploaded.
        workers (int, optional): The maximum number of processes to use for the operation. Default is 8.

    Returns:
        None

    Note:
        This function uploads files from the specified directory to the provided Google Cloud Storage bucket.
        It traverses through the directory recursively, including all files within subdirectories.

    """
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_key
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)

    # Generate a list of paths (in string form) relative to the `source_directory`.
    
    # First, recursively get all files in `source_directory` as Path objects.
    directory_as_path_obj = Path(source_directory)
    paths = directory_as_path_obj.rglob("*")
    
    # Filter so the list only includes files, not directories themselves.
    file_paths = [path for path in paths if path.is_file()]
    
    # These paths are relative to the current working directory. Next, make them
    # relative to `source_directory`.
    relative_paths = [path.relative_to(source_directory) for path in file_paths]

    # Finally, convert them all to strings.
    string_paths = [str(path) for path in relative_paths]

    logging.info("Found {} files.".format(len(string_paths)))
    # print("Found {} files.".format(len(string_paths)))

    # Start the upload.
    results = transfer_manager.upload_many_from_filenames(
        bucket, 
        string_paths, 
        source_directory=source_directory, 
        blob_name_prefix=target_directory, 
        worker_type='thread', 
        max_workers=max_workers, 
        skip_if_exists=True
    )

    for name, result in zip(string_paths, results):
        # The results list is either `None` or an exception for each filename in
        # the input list, in order.
        if isinstance(result, Exception):
            print("Failed to upload {} due to exception: {}".format(name, result))
        else:
            print("Uploaded {} to {}.".format(name, bucket.name))



def upload_file(bucket_name, source_file_name, destination_file_name):
    """
    Uploads a file to a specified bucket in Google Cloud Storage.

    Args:
        bucket_name (str): Name of the bucket to upload the file to.
        source_file_name (str): Local file path of the file to be uploaded.
        destination_file_name (str): Name of the file in the bucket.

    Returns:
        None
    """
    # Instantiating a client for Google Cloud Storage
    storage_client = storage.Client()

    # Getting the specified bucket
    bucket = storage_client.bucket(bucket_name)

    # Creating a blob object within the bucket
    blob = bucket.blob(destination_file_name)

    # Uploading the file to the specified blob in the bucket
    blob.upload_from_filename(source_file_name)


def upload_file_to_folder(destination_blob_name, path_to_file, bucket_name, destination_path=""):
    """
    Uploads a single file to a specified folder within a Google Cloud Storage bucket.

    Args:
        destination_blob_name (str): The name of the file as it will appear in the bucket.
        path_to_file (str): The local path of the file to be uploaded.
        bucket_name (str): The name of the Google Cloud Storage bucket.
        destination_path (str, optional): The folder path within the bucket where the file will be uploaded.
                                           Default is an empty string (root of the bucket).

    Returns:
        None
    """
    # Explicitly use service account credentials by specifying the private key file.
    storage_client = storage.Client()

    # Retrieve the bucket object using the provided bucket name.
    bucket = storage_client.get_bucket(bucket_name)

    # Create a blob object representing the destination within the bucket.
    blob = bucket.blob(destination_path + destination_blob_name)

    # Upload the file from the specified local path to the blob in the bucket.
    blob.upload_from_filename(path_to_file)


# def upload_directory_to_folder(source_directory, gcs_destination):
#     """
#     Uploads files from a local directory to Google Cloud Storage (GCS).

#     Args:
#         source_directory (str): The local directory path containing the files to be uploaded.
#         gcs_destination (str): The destination path in GCS where the files will be uploaded.

#     Returns:
#         None
#     """
#     # Initialize Google Cloud Storage Filesystem
#     fs = gcsfs.GCSFileSystem()

#     # Upload files from local directory to GCS destination
#     fs.put(source_directory, gcs_destination, recursive=True)
