import os
import shutil
from google.cloud import storage
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from mlcore.config.configuration import DataGenerationConfig
# import mlcore.utils.common as utils
from mlcore import logger


class DataGeneration:
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

    def __init__(self, config: DataGenerationConfig):
        self.config = config

    def load_1000_files(self):
        self.move_test_to_train_()
        self.train_test_split()

    def move_test_to_train_(self):
        """Moves the contents of metadata_test.csv file into the 
        metadata_train.csv file by appending to the end of the file.
        The two files are obtained form the train and test directories.
        """
        train_dir = self.config.train_dir
        test_dir = self.config.test_dir
        metadata_train_file = os.path.join(train_dir, 'metadata_train.csv')
        metadata_test_file = os.path.join(test_dir, 'metadata_test.csv')

        # If metadata_test.csv exists
        if os.path.exists(metadata_test_file):
            # Collect previously created test data
            test_metadata_df = pd.read_csv(metadata_test_file)

            # Append the previous test data to the training data
            if os.path.exists(metadata_train_file):
                train_metadata_df = pd.read_csv(metadata_train_file)
                logger.info(f'Found metadata file: {metadata_train_file}')
                train_metadata_df = pd.concat([train_metadata_df, test_metadata_df])
                # Re-write the metadata_train.csv file
                train_metadata_df.to_csv(metadata_train_file, index=False)
                logger.info(f'Updated and saved metadata file: {metadata_train_file}')
                
            # If metadata_train.csv not exists, create it
            else:
                logger.info(f'File not found: {metadata_train_file}')
                test_metadata_df.to_csv(metadata_train_file, index=False)
                logger.info(f'Created new file: {metadata_train_file}')
            
            # Upload metadata_train.csv file to GCP bucket
            DataGeneration.upload_file_to_bucket(bucket_name=self.config.gcp_train_bucket, file_path=metadata_train_file)
            logger.info(f'Uploaded training data to bucket: {metadata_train_file}')
        else:
            logger.info(f'Skipping, file not found: {metadata_test_file}')

    def download_metadata(self):
        # Using an anonymous client since we use a public bucket
        client = storage.Client.create_anonymous_client()
        # Setting our local output directory to store the files
        metadata_dir = self.config.metadata_dir
        metadata_bucket = client.bucket(bucket_name=self.config.gcp_metadata_bucket)
        # List all files present on the bucket
        metadata_files = [blob.name for blob in metadata_bucket.list_blobs()]
        logger.info(f'Metadata files found {metadata_files}')

        # Find the latest metadata file metadata_xx.csv where xx is max
        latest_file_num = max(int(metadata_file.strip('.csv').split('_')[1]) for metadata_file in metadata_files if metadata_file.endswith(".csv"))
        file_name = "metadata_{:02d}.csv".format(latest_file_num)
        logger.info(f'Downloading {file_name} from {metadata_bucket}')

        # Download this file
        blob = metadata_bucket.blob(file_name)
        blob.download_to_filename(os.path.join(metadata_dir, file_name))
        return os.path.join(metadata_dir, file_name), latest_file_num

    @staticmethod
    def upload_file_to_bucket(bucket_name, file_path):
        # Using an anonymous client since we use a public bucket
        client = storage.Client.create_anonymous_client()
        metadata_bucket = client.bucket(bucket_name=bucket_name)
        blob = metadata_bucket.blob(os.path.basename(file_path))
        blob.upload_from_filename(file_path)

    def train_test_split(self):
        current_metadata_file, latest_file_num = self.download_metadata()
        df_metadata = pd.read_csv(current_metadata_file)

        metadata_residual, metadata_test, = train_test_split(df_metadata, test_size=1000, random_state=42, stratify=df_metadata['Emotions'])
        new_metadata_file_name = "metadata_{:02d}.csv".format(latest_file_num + 1)
        new_metadata_file_path = os.path.join(self.config.metadata_dir, Path(new_metadata_file_name))
        metadata_residual.to_csv(new_metadata_file_path, index=False)
        DataGeneration.upload_file_to_bucket(bucket_name=self.config.gcp_metadata_bucket, file_path=new_metadata_file_path)
        
        metadata_test_file = os.path.join(self.config.test_dir, 'metadata_test.csv')
        metadata_test.to_csv(metadata_test_file, index=False)

        # Upload metadata_test.csv file to GCP bucket
        DataGeneration.upload_file_to_bucket(bucket_name=self.config.gcp_test_bucket, file_path=metadata_test_file)
        logger.info(f'Uploaded test data to bucket: {metadata_test_file}')