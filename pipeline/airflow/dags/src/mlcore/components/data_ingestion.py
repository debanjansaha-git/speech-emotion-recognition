import os
import sys
import urllib.request as request
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from google.cloud import storage
import zipfile
from collections import defaultdict 
from mlcore import logger
from mlcore.utils.common import get_size
from pathlib import Path
from mlcore.entity.config_entity import DataIngestionConfig
from tempfile import NamedTemporaryFile

CHUNK_SIZE = 40960


class DataIngestion:
    """
    Class for data ingestion.

    Summary:
        This class handles the downloading and extraction of data from a specified source URL.

    Explanation:
        The DataIngestion class provides a method, download_data(), which downloads and extracts data from a specified source URL.
        If the local data path is empty, the download_data() method retrieves the data from the source URL, saves it to the local data path,
        and extracts the data if it is compressed. If the local data path is not empty, the method skips the download process.
        The class takes a DataIngestionConfig object as input, which contains the necessary configuration parameters for data ingestion.

    Args:
        config (DataIngestionConfig): The configuration object containing the necessary parameters for data ingestion.

    Methods:
        download_data(): Downloads and extracts data from the specified source URL.

    Raises: HTTPError: If there is an error while downloading the data from the source URL. OSError: If there is an error while saving the data to the local data path.

    Examples: data_ingestion = DataIngestion(config) data_ingestion.download_data()
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        """
        Function for downloading data.

        Summary:
            This function handles the downloading of data from a specified source URL.

        Explanation:
            The download_data() function downloads data from a specified source URL and saves it to the local data path.
            If the local data path is not empty, the function skips the download process.

        Args:
            self: The instance of the class.

        Returns:
            None.

        Raises:
            None.
        """

        if len(os.listdir(self.config.local_data_path)) == 0:
            directory, download_url_encoded = self.config.source_URL.split(":")
            download_url = unquote(download_url_encoded)
            filename = urlparse(download_url).path
            destination_path = os.path.join(self.config.root_dir, directory)
            try:
                with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
                    total_length = fileres.headers["content-length"]
                    logger.info(
                        f"Downloading {directory}, {total_length} bytes compressed"
                    )
                    dl = 0
                    data = fileres.read(CHUNK_SIZE)
                    while len(data) > 0:
                        dl += len(data)
                        tfile.write(data)
                        done = int(50 * dl / int(total_length))
                        sys.stdout.write(
                            f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded"
                        )
                        sys.stdout.flush()
                        data = fileres.read(CHUNK_SIZE)
                    logger.info("\n")
                    if filename.endswith(".zip"):
                        logger.info(f"Uncompressing {filename}")
                        with zipfile.ZipFile(tfile) as zfile:
                            zfile.extractall(destination_path)

                    else:
                        logger.info(f"Uncompressing {filename}")
                        with tarfile.open(tfile.name) as tarfile:
                            tarfile.extractall(destination_path)
                    logger.info(f"Downloaded and uncompressed: {directory}")
            except HTTPError as e:
                logger.error(
                    f"Failed to load (likely expired) {download_url} to path {destination_path}"
                )
            except OSError as e:
                logger.error(
                    f"Failed to load {download_url} to path {destination_path}"
                )
        else:
            logger.info(
                f"Data already exists in location {self.config.local_data_path}, skipping download!"
            )

    def download_from_gcp(self, threads=8):
        """Downloads data from a public google cloud bucket. Uses input and 
        output paths from a config file with the help of the config attribute.

        Args:
            threads (int, optional): Number of thread to use for parallel downloads. Defaults to 8.
        """
        
        # Using an anonymous client since we use a public bucket
        client = storage.Client.create_anonymous_client()
        # Setting our local output directory to store the files
        # TODO: Use the artifacts root instead of 'data_ingestion/' dir
        output_dir = Path(self.config.root_dir.replace('data_ingestion', ''))
        bucket = client.bucket(bucket_name=self.config.gcp_bucket_name)
        # List all files present on the bucket
        file_paths = [blob.name for blob in bucket.list_blobs(prefix=str(self.config.gcp_data_path))]
        
        result_dict = defaultdict(lambda: 0)
        files_to_download = []
        for file_path in file_paths:
            p = Path(file_path)
            if os.path.exists(os.path.join(output_dir, p)):
                result_dict['skipped'] += 1
            else:
                files_to_download.append(file_path)
        result_dict['files_to_download'] = len(files_to_download)
        
        logger.info(f'Starting Download for {len(files_to_download)} files')
        # Download the files using multiple threads
        results = storage.transfer_manager.download_many_to_path(
            bucket, files_to_download, 
            destination_directory=output_dir, 
            worker_type=storage.transfer_manager.THREAD, 
            max_workers=threads
        )

        # Observe the result status for all the files
        for name, result in zip(files_to_download, results):
        # The results list is either `None` or an exception for each blob in
        # the input list, in order.
            if isinstance(result, Exception):
                result_dict['failed'] += 1
            else:
                result_dict['downloaded'] += 1
        logger.info(f'Download complete. Results: {result_dict.items()}')