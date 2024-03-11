from google.cloud import storage
from pathlib import Path
from google.cloud.storage import Client, transfer_manager
import logging
import gcsfs

# Upload only to a Bucket
# upload a single file
def upload_file(bucket_name, source_file_name, destination_file_name): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_file_name)
    blob.upload_from_filename(source_file_name)
# Sample usage
# upload_file('test_demo_storage_bucket_venky', '/home/venky/imperial-ally-416204-e6815a4b11d5.json', 'test.json')

# Upload a whole directory
def upload_directory(bucket_name, source_directory, workers=8):
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

    logging.debug("Found {} files.".format(len(string_paths)))
    # print("Found {} files.".format(len(string_paths)))

    # Start the upload.
    results = transfer_manager.upload_many_from_filenames(
        bucket, string_paths, source_directory=source_directory, max_workers=workers
    )

    for name, result in zip(string_paths, results):
        # The results list is either `None` or an exception for each filename in
        # the input list, in order.
        if isinstance(result, Exception):
            print("Failed to upload {} due to exception: {}".format(name, result))
        else:
            print("Uploaded {} to {}.".format(name, bucket.name))

# Sample usage
# upload_directory_with_transfer_manager('test_demo_storage_bucket_venky/test1', '/home/venky/Audio_Song_Actors_01-24',workers=8)


# Upload to a particular folder in GCS Bucket
            

# upload a single file to a folder
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

## Sample usage
# upload_file_to_folder('test_in_gcs.txt', 'test.txt', 'test_demo_storage_bucket_venky', 'test1/')
    
# upload a directory to a folder
def upload_directory_to_folder(source_directory, gcs_destination):
    """
    Uploads files from a local directory to Google Cloud Storage (GCS).

    Parameters:
    - src_dir (str): The local directory path containing the files to be uploaded.
    - gcs_dst (str): The destination path in GCS where the files will be uploaded.

    Returns:
    - None

    Example:
    upload_to_gcs('/path/to/local/directory', 'gs://bucket-name/destination_folder')
    """

    # Initialize Google Cloud Storage Filesystem
    fs = gcsfs.GCSFileSystem()

    # Upload files from local directory to GCS destination
    fs.put(source_directory, gcs_destination, recursive=True)

## Sample usage
# upload_directory_to_folder('/home/venky/Audio_Song_Actors_01-24', "test_demo_storage_bucket_venky/my_folder")