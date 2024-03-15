import io
import os
import warnings
import yaml
import timeit
import multiprocessing as mp
from joblib import Parallel, delayed
import cProfile
import pstats

import numpy as np
import pandas as pd
import librosa
from librosa.feature import melspectrogram, mfcc
from sklearn.model_selection import train_test_split

from mlcore import logger
from mlcore.constants import *
from mlcore.entity.config_entity import DataTransformationConfig

warnings.filterwarnings("ignore")


# For reproducibility
np.random.seed(42)


class AudioAugmenter:
    """
    Class for audio data augmentation.

    Summary:
        This class provides methods to augment audio data by adding noise, time-stretching, shifting, and changing pitch.

    Explanation:
        The AudioAugmenter class contains methods to apply various augmentation techniques to audio data.
        The noise() method adds additive white Gaussian noise (AWGN) to the audio data, providing a more realistic simulation of background noise or environmental conditions.
        The stretch() method time-stretches the audio data by a specified rate using the time_stretch function.
        The shift() method shifts the audio data by a random amount within a certain range, simulating changes in timing or alignment.
        The pitch() method changes the pitch of the audio data by a specified number of steps using the pitch_shift function.
        The class takes an optional noise_std parameter, which controls the standard deviation of the Gaussian noise added in the noise() method.

    Methods:
        noise(data: np.ndarray) -> np.ndarray:
            Adds additive white Gaussian noise (AWGN) to the audio data and returns the augmented data.

        stretch(data: np.ndarray, rate: float = 0.8) -> np.ndarray:
            Time-stretches the audio data by the specified rate and returns the augmented data.

        shift(data: np.ndarray) -> np.ndarray:
            Shifts the audio data by a random amount within a certain range and returns the augmented data.

        pitch(data: np.ndarray, sampling_rate: int, n_steps: int = 3) -> np.ndarray:
            Changes the pitch of the audio data by the specified number of steps and returns the augmented data.

    Args:
        data (np.ndarray): The input audio data.
        rate (float, optional): The rate of time-stretching. Default is 0.8.
        sampling_rate (int): The sampling rate of the audio data.
        n_steps (int, optional): The number of steps to change the pitch. Default is 3.

    Returns:
        np.ndarray: The augmented audio data.

    Examples:
        augmenter = AudioAugmenter()
        augmented_data = augmenter.noise(data)
    """

    def __init__(self, noise_std=0.035):
        self.noise_std = noise_std

    # NOISE
    def noise(self, data):
        """This method adds additive white Gaussian noise (AWGN) to the audio data.
        Gaussian noise can provide a more realistic simulation of background noise or environmental conditions that may affect speech signals
        """
        # amplitude-dependent additive noise
        # noise_amp = 0.035 * np.random.uniform() * np.amax(data)
        # data = data + noise_amp * np.random.normal(size=data.shape[0])
        noise = np.random.normal(scale=self.noise_std, size=data.shape[0])
        return data + noise

    # STRETCH
    def stretch(self, data, rate=0.8):
        """This method time-stretches the audio data by a specified rate.
        It uses the time_stretch function to perform time stretching with the specified rate.
        """
        return librosa.effects.time_stretch(data, rate=0.8)

    # SHIFT
    def shift(self, data):
        """This method shifts the audio data by a random amount within a certain range.
        It generates a random shift range using a uniform distribution and then rolls (shifts) the audio data by the generated amount.
        """
        shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
        return np.roll(data, shift_range)

    # PITCH
    def pitch(self, data, sampling_rate, n_steps=3):
        """This method changes the pitch of the audio data by a specified number of steps.
        It uses the pitch_shift function to shift the pitch of the audio data by the specified number of steps.
        """
        return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=n_steps)


class FeatureExtractor:
    """
    Class for feature extraction.

    Summary:
        This class provides methods to extract various audio features from input data.

    Explanation:
        The FeatureExtractor class contains methods to extract features such as zero-crossing rate (ZCR), root mean square energy (RMSE),
        Mel-frequency cepstral coefficients (MFCC), chromagram, Mel Spectrogram from audio data. The extract_features() method combines these features into a single array
        and returns the result. The class takes optional parameters for frame length and hop length, which control the size and overlap of the analysis windows.

    Methods:
        extract_features(data: np.ndarray, sr: int = 22050) -> np.ndarray:
            Extracts audio features from the input data and returns the combined feature array.

    Private Methods:
        __zcr__(data: np.ndarray) -> np.ndarray:
            Calculates the zero-crossing rate (ZCR) feature from the input data and returns the result.

        __rmse__(data: np.ndarray) -> np.ndarray:
            Calculates the root mean square energy (RMSE) feature from the input data and returns the result.

        __chroma__(data: np.ndarray, sr: int) -> np.ndarray:
            Calculates chromagram or power spectrogram from the input data and returns the result.

        __melspec__(data: np.ndarray, sr: int) -> np.ndarray:
            Calculates Mel Spectrogram from the input data and returns the result.

        __mfcc__(data: np.ndarray, sr: int, flatten: bool = True) -> np.ndarray:
            Calculates the Mel-frequency cepstral coefficients (MFCC) feature from the input data and returns the result.
            The flatten parameter determines whether to flatten the MFCC array or not. Default is True.

    Args:
        data (np.ndarray): The input audio data.
        sr (int, optional): The sample rate of the audio data. Default is 22050.

    Returns:
        np.ndarray: The combined feature array extracted from the input data.

    Examples:
        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(data)
    """

    def __init__(self, frame_length=2048, hop_length=512):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def extract_features(self, data, sr=22050):
        """Extracts audio features from the input data and returns the combined feature array."""
        result = np.array([])
        # Extracting features: ZCR, RMSE, and MFCC
        result = np.hstack(
            (
                result,
                self.__zcr__(data),
                self.__rmse__(data),
                self.__chroma__(data, sr),
                self.__melspec__(data, sr),
                self.__mfcc__(data, sr),
            )
        )
        return result

    def __zcr__(self, data):
        """Calculates the zero-crossing rate (ZCR) feature from the input data and returns the result."""
        zcr = np.mean(
            librosa.feature.zero_crossing_rate(
                data, frame_length=self.frame_length, hop_length=self.hop_length
            ).T,
            axis=0,
        )
        return np.squeeze(zcr)

    def __rmse__(self, data):
        """Calculates the root mean square energy (RMSE) feature from the input data and returns the result."""
        rmse = np.mean(
            librosa.feature.rms(
                y=data, frame_length=self.frame_length, hop_length=self.hop_length
            ).T,
            axis=0,
        )
        return np.squeeze(rmse)

    def __chroma__(self, data, sr):
        """Calculates chromagram or power spectrogram from the input data and returns the result."""
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(
            librosa.feature.chroma_stft(S=stft, sr=sr, hop_length=self.hop_length).T,
            axis=0,
        )
        return np.squeeze(chroma_stft)

    def __melspec__(self, data, sr):
        """Calculates Mel Spectrogram from the input data and returns the result."""
        spectro = np.mean(
            librosa.feature.melspectrogram(y=data, sr=sr, hop_length=self.hop_length).T,
            axis=0,
        )
        return np.squeeze(spectro)

    def __mfcc__(self, data, sr, flatten=True):
        """Calculates the Mel-frequency cepstral coefficients (MFCC) feature from the input data and returns the result.
        The flatten parameter determines whether to flatten the MFCC array or not. Default is True.
        """
        mfccs = np.mean(
            librosa.feature.mfcc(
                y=data,
                sr=sr,
                n_mfcc=128,
                n_fft=self.frame_length,
                hop_length=self.hop_length,
            ).T,
            axis=0,
        )
        return np.ravel(mfccs) if flatten else np.squeeze(mfccs)


class DataTransformation:
    """
    Class for data transformation.

    Summary:
        This class handles the transformation of audio data by applying various augmentation techniques and extracting features.

    Explanation:
        The DataTransformation class provides methods to transform audio data by adding noise, time-stretching, shifting, and changing pitch.
        It also includes methods for feature extraction from the transformed audio data.
        The class takes a DataTransformationConfig object as input, which contains the necessary configuration parameters for data transformation.
        The get_features() method loads audio data from a specified path and extracts features using the FeatureExtractor class.
        The process_feature() method processes a single audio file by extracting features and associating them with a specified emotion label.
        The feature_engineering() method performs data transformation and feature extraction on a dataset in parallel using multiprocessing.
        The train_test_split_data() method splits the transformed dataset into train and test sets.
        Transforms the data by augmenting it with standard audio augmentation transforms

        Justification:
        https://aws.amazon.com/what-is/data-augmentation/
        Audio transformations typically include injecting random or Gaussian noise into some audio,
        fast-forwarding parts, changing the speed of parts by a fixed rate, or altering the pitch.

    Args:
        config (DataTransformationConfig): The configuration object containing the necessary parameters for data transformation.

    Methods:
        get_features(path: str, duration: float = 2.5, offset: float = 0.6) -> np.ndarray:
            Loads audio data from the specified path, extracts features, and returns the feature array.

        process_feature(path: str, emotion: str) -> Tuple[List[np.ndarray], List[str]]:
            Processes a single audio file by extracting features and associating them with the specified emotion label.
            Returns the feature array and emotion labels.

        feature_engineering():
            Performs data transformation and feature extraction on the dataset in parallel using multiprocessing.

        train_test_split_data(test_size: float = 0.2):
            Splits the transformed dataset into train and test sets and saves them to disk.

    Raises:
        No transformation parameters specified: If no transformation parameters are specified in the configuration.

    Examples:
        data_transformation = DataTransformation(config)
        features = data_transformation.get_features(path)
        X, Y = data_transformation.process_feature(path, emotion)
        data_transformation.feature_engineering()
        data_transformation.train_test_split_data(test_size)
    """

    def __init__(self, config: DataTransformationConfig):
        """
        Class for data transformation.

        Summary:
            This class handles the transformation of data using audio augmentation and feature extraction techniques.

        Explanation:
            The DataTransformation class takes a DataTransformationConfig object as input and provides methods for audio augmentation and feature extraction.
            It initializes the AudioAugmenter and FeatureExtractor classes and uses the specified configuration parameters for data transformation.

        Args:
            config (DataTransformationConfig): The configuration object containing the necessary parameters for data transformation.

        Methods:
            None.

        Raises:
            None.
        """

        self.config = config
        self.chunksize = 1000  # for processing data in chunks
        # Audio Augmentation & Feature Extraction
        self.aug = AudioAugmenter()
        self.feat = FeatureExtractor()
        # read data augmentation params from config file
        # option to try multiple augmentation params and observe the influence on model performance
        with open(PARAMS_FILE_PATH, "r") as f:
            tfx_params = yaml.safe_load(f)
        self.tfx_params = tfx_params["data_transforms"]["params"]

    def get_features(self, path, duration=2.5, offset=0.6):
        """
        Function for extracting features from audio data.

        Summary:
            This function extracts features from audio data using various transformation techniques.

        Explanation:
            The get_features() function takes an audio file path as input and extracts features from the audio data.
            It applies different transformation techniques such as adding noise, time stretching, and pitch shifting to the audio data.
            The function returns a feature array containing the extracted features.

        Args:
            path (str): The path to the audio file.
            duration (float): The duration of the audio segment to consider (default: 2.5 seconds).
            offset (float): The offset from the beginning of the audio file to start the segment (default: 0.6 seconds).

        Returns:
            np.ndarray: The feature array containing the extracted features.

        Raises:
            None.
        """

        # Load raw audio data
        data, sr = librosa.load(path, duration=duration, offset=offset)
        audio_feats = []
        # perform data augmentation
        for param in self.tfx_params:
            if param == "default":
                audio_feats.append(self.feat.extract_features(data))
            elif param == "noise":
                audio_feats.append(self.feat.extract_features(self.aug.noise(data)))
            elif param == "pitch":
                audio_feats.append(self.feat.extract_features(self.aug.pitch(data, sr)))
            elif param == "pitch_noise":
                pitch_audio = self.aug.pitch(data, sr)
                audio_feats.append(
                    self.feat.extract_features(self.aug.noise(pitch_audio))
                )
            elif param == "pitch_shift_noise":
                pitch_audio = self.aug.pitch(data, sr)
                shift_audio = self.aug.shift(pitch_audio)
                audio_feats.append(
                    self.feat.extract_features(self.aug.noise(shift_audio))
                )
            elif param == "pitch_shift_stretch_noise":
                pitch_audio = self.aug.pitch(data, sr)
                shift_audio = self.aug.shift(pitch_audio)
                stretch_audio = self.aug.stretch(shift_audio)
                audio_feats.append(
                    self.feat.extract_features(self.aug.noise(stretch_audio))
                )
            elif param == "shift_noise":
                shift_audio = self.aug.shift(data)
                audio_feats.append(
                    self.feat.extract_features(self.aug.noise(shift_audio))
                )
            elif param == "stretch":
                audio_feats.append(self.feat.extract_features(self.aug.stretch(data)))
            elif param == "stretch_noise":
                stretch_audio = self.aug.stretch(data)
                audio_feats.append(
                    self.feat.extract_features(self.aug.noise(stretch_audio))
                )
            elif param == "stretch_pitch_noise":
                pitch_audio = self.aug.pitch(data, sr)
                stretch_audio = self.aug.stretch(pitch_audio)
                audio_feats.append(
                    self.feat.extract_features(self.aug.noise(stretch_audio))
                )
            elif param == "stretch_shift_noise":
                stretch_audio = self.aug.stretch(data)
                shift_audio = self.aug.shift(stretch_audio)
                audio_feats.append(
                    self.feat.extract_features(self.aug.noise(shift_audio))
                )
            else:
                logger.error("No transformation parameters specified!")
        # stack and return augmented audio representing real world scenario
        audio = np.vstack(audio_feats)
        return audio

    def process_feature(self, path, emotion):
        """
        Function for processing a single audio feature.

        Summary:
            This function processes a single audio file by extracting features and associating them with a specified emotion label.

        Explanation:
            The process_feature() function takes an audio file path and an emotion label as input.
            It calls the get_features() function to extract features from the audio file and associates them with the specified emotion label.
            The function returns the feature array and emotion labels.

        Args:
            path (str): The path to the audio file.
            emotion (str): The emotion label associated with the audio file.

        Returns:
            Tuple[List[np.ndarray], List[str]]: The feature array and emotion labels.

        Raises:
            None.
        """

        features = self.get_features(path)
        X = []
        Y = []
        for ele in features:
            X.append(ele)
            Y.append(emotion)
        return X, Y

    def feature_engineering(self):
        """
        Function for feature engineering.

        Summary:
            This function performs feature engineering on audio data.

        Explanation:
            The feature_engineering() function takes the root directory and metadata path as input.
            It reads the metadata file, drops any rows with missing values, and extracts features from the audio data.
            The function saves the feature array to disk and returns descriptive statistics of the data.

        Args:
            root_dir (str): The root directory path.
            metadata_dir (str): The path to the metadata file.

        Returns:
            None.

        Raises:
            None.
        """

        root_dir = self.config.root_dir
        metadata_dir = self.config.metadata_path
        profiler = cProfile.Profile()
        streams = io.StringIO()

        data = pd.read_csv(metadata_dir)
        data = data.dropna()
        paths = data["FilePath"]
        emotions = data["Emotions"]
        logger.warning(" !!!! OVERCLOCK ALERT !!!! ")
        logger.warning(" ==== Using Multiprocessors for Data Transformation ====")
        num_cpus = mp.cpu_count() - 1  # leave atleast 1 CPU core for other tasks
        logger.info(f"Number of processors: {str(num_cpus)}")
        start = timeit.default_timer()
        logger.info("Multiprocessing started!")
        # Perform feature extraction in parallel using multiprocessing
        profiler.enable()  # enable cProfiler for monitoring
        results = Parallel(n_jobs=-2)(
            delayed(self.process_feature)(path, emotion)
            for (path, emotion) in zip(paths, emotions)
        )
        profiler.disable()  # terminate profiler
        elapsed_time = timeit.default_timer() - start
        logger.info(f"Elapsed Time: {elapsed_time:.2f} secs")
        # retrieve top 20 logs in accordance to total time spent descending
        stats = (
            pstats.Stats(profiler, stream=streams).sort_stats("tottime").print_stats(20)
        )
        logger.info(f"Stats from Multiprocessing:\n{streams.getvalue()}")

        logger.info("Trying to export dataset to disk....")
        start = timeit.default_timer()
        # Start processing data in chunks of self.chunksize to reduce IO overheads
        profiler.enable()
        for i, chunk_start in enumerate(range(0, len(results), self.chunksize)):
            logger.info(f"Processing Chunk {i} now...")
            chunk_end = min(chunk_start + self.chunksize, len(results))
            results_chunk = results[chunk_start:chunk_end]

            X_chunk = []
            Y_chunk = []
            # Unravel features (ndarrays) before creating dataframe
            for result in results_chunk:
                x, y = result
                X_chunk.extend(x)
                Y_chunk.extend(y)

            emotions_df = pd.DataFrame(X_chunk)
            emotions_df.fillna(
                0, inplace=True
            )  # fill missing value w/ 0 - short duration audio
            emotions_df["Emotions"] = Y_chunk  # Extract labels

            # Converted unstructured data -> structured data & storing as parquet for later use
            emotions_df.to_parquet(
                f"{self.config.output_path}/data_part_{i}.parquet", compression="gzip"
            )
        profiler.disable()
        stats = (
            pstats.Stats(profiler, stream=streams).sort_stats("tottime").print_stats(20)
        )
        logger.info(f"Stats from WriteParquet:\n{streams.getvalue()}")

        logger.info("Dataframe written to disk!!")
        elapsed_time = timeit.default_timer() - start
        logger.info(f"Elapsed Time: {elapsed_time:.2f} secs")

    def get_data_paths(self, output_path):
        """
        Checks if the output directory exists and returns a list of data file paths matching the pattern.

        Args:
            output_path (str): The path to the directory containing data files.

        Returns:
            list: A list of data file paths.

        Raises:
            FileNotFoundError: If the output directory does not exist or no data files are found.
        """
        if not os.path.isdir(output_path):
            raise FileNotFoundError(f"Output directory {output_path} does not exist!")

        data_files = os.listdir(output_path)
        if data_paths := [
            os.path.join(output_path, filename)
            for filename in data_files
            if filename.startswith("data_part_") and filename.endswith(".parquet")
        ]:
            return data_paths

        else:
            raise FileNotFoundError("No data files found matching the pattern!")

    def train_test_split_data(self, test_size=0.2, val_size=0.5):
        """
        Splits the data into train, validation, and test sets.

        Args:
            test_size (float, optional): The proportion of the data to use for the test set. Defaults to 0.2.
            val_size (float, optional): The proportion of the data to use for the validation set. Defaults to 0.5.

        Returns:
            tuple: A tuple containing the train, validation, and test dataframes.
        """
        data_paths = self.get_data_paths(self.config.output_path)
        all_data = [pd.read_parquet(path) for path in data_paths]
        emotions_df = pd.concat(all_data, ignore_index=True)

        train, test = train_test_split(
            emotions_df, test_size=test_size, random_state=42
        )
        test, val = train_test_split(test, test_size=val_size, random_state=42)
        logger.info(
            f"Shapes ==> Train: {train.shape}, Val: {val.shape}, Test: {test.shape}"
        )
        return train, val, test

    def split_and_scale(self, test_size, val_size, method="train"):
        """
        Splits the data into train, validation, and test sets, and scales the features.

        Args:
            method (str, optional): The method to use for splitting and scaling. Defaults to "train".
            test_size (float, optional): The proportion of the data to use for the test set. Defaults to 0.2.
            val_size (float, optional): The proportion of the data to use for the validation set. Defaults to 0.5.

        Raises:
            RuntimeError: If `split_and_scale` is called with `method='test'` before calling it with `method='train'`.

        Examples:
            # Example 1: Split and scale the data for training
            split_and_scale(method="train", test_size=0.2, val_size=0.5)

            # Example 2: Scale the test data using the previously computed mean and standard deviation
            split_and_scale(method="test")
        """

        if method == "train":
            train_data, val_data, test_data = self.train_test_split_data(
                test_size=test_size, val_size=val_size
            )
            X_train = train_data.drop("Emotions", axis=1)
            X_val = val_data.drop("Emotions", axis=1)
            y_train = train_data["Emotions"]
            y_val = val_data["Emotions"]
            self.X_mean = np.mean(X_train, axis=0)
            self.X_std = np.std(X_train, axis=0)
            X_train = (X_train - self.X_mean) / self.X_std
            X_val = (X_val - self.X_mean) / self.X_std
            X_train["Emotions"] = y_train
            X_val["Emotions"] = y_val
            X_train.to_parquet(self.config.train_path, compression="gzip")
            X_val.to_parquet(self.config.val_path, compression="gzip")

        elif method == "test":
            if not hasattr(self, "X_mean") or not hasattr(self, "X_std"):
                logger.error("Call the split_and_scale(method='train') first!!!")
                raise RuntimeError("Call the split_and_scale(method='train') first!!!")
            else:
                X_test = test_data.drop("Emotions", axis=1)
                y_test = test_data["Emotions"]
                X_test = (X_test - self.X_mean) / self.X_std
                X_test["Emotions"] = y_test
                X_test.to_parquet(self.config.test_path, compression="gzip")

        logger.info("Dataframe written to disk!!")
