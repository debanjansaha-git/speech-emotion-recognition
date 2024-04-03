import os
import shutil
from google.cloud import storage
import pandas as pd
from sklearn.model_selection import train_test_split
from mlcore import logger
from pathlib import Path
from mlcore.entity.config_entity import DataGenerationConfig
import mlcore.utils.common as utils


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
        self.move_test_to_train()
        self.train_test_split()

    def move_test_to_train(self):
        logger.info(f'Train Dir: {self.config.train_dir}')
        logger.info(f'Test Dir: {self.config.test_dir}')
        train_dir = self.config.train_dir
        test_dir = self.config.test_dir
        test_size_before = len(os.listdir(test_dir))
        train_size_before = len(os.listdir(test_dir))
        
        utils.move_files(source=test_dir, destination=train_dir)
        try:
            train_metadata_df = pd.read_csv(os.path.join(train_dir, 'metadata_train.csv'))
            test_metadata_df = pd.read_csv(os.path.join(train_dir, 'metadata_test.csv'))
            new_train_metadata_df = pd.concat([train_metadata_df, test_metadata_df])
            new_train_metadata_df.to_csv(os.path.join(train_dir, 'metadata_train.csv'), index=False)
        except FileNotFoundError:
            try:
                logger.info(f"Missing {os.path.join(train_dir, 'metadata_train.csv')}. Is your train folder empty?")
                test_metadata_df = pd.read_csv(os.path.join(train_dir, 'metadata_test.csv'))
                test_metadata_df.to_csv(os.path.join(train_dir, 'metadata_train.csv'), index=False)
            except FileNotFoundError:
                logger.info(f"Missing {os.path.join(train_dir, 'metadata_train.csv')} and {os.path.join(train_dir, 'metadata_test.csv')}. Are your train and test folders empty?")
            
        test_size_after = len(os.listdir(test_dir))
        train_size_after = len(os.listdir(test_dir))
        if test_size_after == 0:
            if train_size_after == (train_size_before + (test_size_before - test_size_after)):
                logger.info(f'Moved {test_size_before - test_size_after} Files from Test to Train Folder')
        else:
            logger.info(f'Test Before: {test_size_before}, Test After: {test_size_after}')
            logger.info(f'Train Before: {train_size_before}, Train After: {train_size_after}')

    def download_metadata(self):
        # Using an anonymous client since we use a public bucket
        client = storage.Client.create_anonymous_client()
        # Setting our local output directory to store the files
        metadata_dir = self.config.metadata_dir
        metadata_bucket = client.bucket(bucket_name=self.config.gcp_metadata_bucket)
        # List all files present on the bucket
        metadata_files = [blob.name for blob in metadata_bucket.list_blobs()]
        logger.info(f'Metadata file found {metadata_files}')

        # Find the latest metadata file metadata_xx.csv where xx is max
        latest_file_num = max(int(metadata_file.strip('.csv').split('_')[1]) for metadata_file in metadata_files if metadata_file.endswith(".csv"))
        file_name = "metadata_{:02d}.csv".format(latest_file_num)
        logger.info(f'Downloading {file_name} from {metadata_bucket}')

        # Download this file
        blob = metadata_bucket.blob(file_name)
        blob.download_to_filename(os.path.join(metadata_dir, file_name))
        return os.path.join(metadata_dir, file_name), latest_file_num

    def upload_metadata(self, file_path):
        # Using an anonymous client since we use a public bucket
        client = storage.Client.create_anonymous_client()
        metadata_bucket = client.bucket(bucket_name=self.config.gcp_metadata_bucket)
        blob = metadata_bucket.blob(os.path.basename(file_path))
        blob.upload_from_filename(file_path)

    def train_test_split(self):
        current_metadata_file, latest_file_num = self.download_metadata()
        df_metadata = pd.read_csv(current_metadata_file)

        metadata_train, metadata_test, = train_test_split(df_metadata, test_size=1000, random_state=42, stratify=df_metadata['Emotions'])
        new_metadata_file_name = "metadata_{:02d}.csv".format(latest_file_num + 1)
        new_metadata_file_path = os.path.join(self.config.metadata_dir, Path(new_metadata_file_name))
        metadata_train.to_csv(new_metadata_file_path, index=False)
        self.upload_metadata(file_path=new_metadata_file_path)
        self.copy_files_to_test_dir(metadata_test['FilePath'].values)
        metadata_test.to_csv(os.path.join(self.config.test_dir, 'metadata_test.csv'), index=False)

    def copy_files_to_test_dir(self, files_to_copy):
        count = 0
        for idx, path in enumerate(files_to_copy):
            shutil.copyfile(path, os.path.join(self.config.test_dir, os.path.basename(path)))
            count = idx + 1
        logger.info(f'Copied {count} files from Dataset to Test Folder')
        
    
