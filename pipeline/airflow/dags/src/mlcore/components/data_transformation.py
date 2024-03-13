import os
import numpy as np
import pandas as pd
import yaml
import timeit
from sklearn.model_selection import train_test_split
from mlcore import logger
from mlcore.constants import *
from mlcore.entity.config_entity import DataTransformationConfig
import librosa
from librosa.feature import melspectrogram, mfcc
import multiprocessing as mp
from joblib import Parallel, delayed

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
        and Mel-frequency cepstral coefficients (MFCC) from audio data. The extract_features() method combines these features into a single array
        and returns the result. The class takes optional parameters for frame length and hop length, which control the size and overlap of the analysis windows.

    Methods:
        extract_features(data: np.ndarray, sr: int = 22050) -> np.ndarray:
            Extracts audio features from the input data and returns the combined feature array.

    Private Methods:
        __zcr__(data: np.ndarray) -> np.ndarray:
            Calculates the zero-crossing rate (ZCR) feature from the input data and returns the result.

        __rmse__(data: np.ndarray) -> np.ndarray:
            Calculates the root mean square energy (RMSE) feature from the input data and returns the result.

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
        result = np.array([])

        result = np.hstack(
            (
                result,
                self.__zcr__(data),
                self.__rmse__(data),
                self.__mfcc__(data, sr),
            )
        )
        return result

    def __zcr__(self, data):
        zcr = librosa.feature.zero_crossing_rate(
            data, frame_length=self.frame_length, hop_length=self.hop_length
        )
        return np.squeeze(zcr)

    def __rmse__(self, data):
        rmse = librosa.feature.rms(
            y=data, frame_length=self.frame_length, hop_length=self.hop_length
        )
        return np.squeeze(rmse)

    def __mfcc__(self, data, sr, flatten=True):
        mfccs = librosa.feature.mfcc(
            y=data,
            sr=sr,
            n_mfcc=20,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        return np.squeeze(mfccs.T) if not flatten else np.ravel(mfccs.T)


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
        self.config = config
        # Audio Augmentation & Feature Extraction
        self.aug = AudioAugmenter()
        self.feat = FeatureExtractor()

        with open(PARAMS_FILE_PATH, "r") as f:
            tfx_params = yaml.safe_load(f)
        self.tfx_params = tfx_params["data_transforms"]["params"]

    def get_features(self, path, duration=2.5, offset=0.6):

        # Load data
        data, sr = librosa.load(path, duration=duration, offset=offset)
        data_tuple = ()
        for param in self.tfx_params:
            if param == "default":
                # Extract features from the audio signal
                og_audio = self.feat.extract_features(data)
                default_feat = np.array(og_audio)
                # logger.info("Extracting features from original audio")
            ### Augmented Features ###
            elif param == "noise":
                # Add AGWN
                nz_audio = self.aug.noise(data)
                noise_feat = self.feat.extract_features(nz_audio)
                # logger.info("Adding AGWN transforms to original audio")
            elif param == "stretch":
                # Add time strech
                str_audio = self.aug.stretch(data)
                stretch_feat = self.feat.extract_features(str_audio)
                # logger.info("Adding time stretch transforms to original audio")
            elif param == "pitch":
                # Pitch shift
                pch_audio = self.aug.pitch(data, sr)
                pitch_feat = self.feat.extract_features(pch_audio)
                # logger.info("Adding pitch shift to original audio")
            ### Polynomial Feature Augmentation ###
            elif param == "pitch_noise":
                # Add AGWN to pitch shifted audio
                imd_pch_audio = self.aug.pitch(data, sr)
                pch_nz_audio = self.aug.noise(imd_pch_audio)
                pitch_noise_feat = self.feat.extract_features(pch_nz_audio)
                # logger.info("Adding AGWN to pitch shifted audio")
            elif param == "stretch_noise":
                # Add AGWN to pitch shifted audio
                imd_str_audio = self.aug.stretch(data)
                str_nz_audio = self.aug.noise(imd_str_audio)
                stretch_noise_feat = self.feat.extract_features(str_nz_audio)
                # logger.info("Adding AGWN to time streched audio")
            elif param == "shift_noise":
                # Add AGWN to pitch shifted audio
                imd_sht_audio = self.aug.shift(data)
                sht_nz_audio = self.aug.noise(imd_sht_audio)
                shift_noise_feat = self.feat.extract_features(sht_nz_audio)
                # logger.info("Adding AGWN to shifted audio")
            elif param == "stretch_shift_noise":
                imd_str_audio = self.aug.stretch(data)
                imd_ss_audio = self.aug.shift(imd_str_audio)
                ss_nz_audio = self.aug.noise(imd_ss_audio)
                stretch_shift_noise_feat = self.feat.extract_features(ss_nz_audio)
                # logger.info("Adding AGWN to strech-shifted audio")
            elif param == "stretch_pitch_noise":
                imd_pch_audio = self.aug.pitch(data, sr)
                imd_sp_audio = self.aug.stretch(imd_pch_audio)
                sp_nz_audio = self.aug.noise(imd_sp_audio)
                stretch_pitch_noise_feat = self.feat.extract_features(sp_nz_audio)
                # logger.info("Adding AGWN to strech-pitched audio")
            elif param == "pitch_shift_noise":
                imd_pch_audio = self.aug.pitch(data, sr)
                imd_ps_audio = self.aug.shift(imd_pch_audio)
                ps_nz_audio = self.aug.noise(imd_ps_audio)
                pitch_shift_noise_feat = self.feat.extract_features(ps_nz_audio)
                # logger.info("Adding AGWN to pitch-shifted audio")
            elif param == "pitch_shift_stretch_noise":
                imd_pch_audio = self.aug.pitch(data, sr)
                imd_ps_audio = self.aug.shift(imd_pch_audio)
                imd_pss_audio = self.aug.stretch(imd_ps_audio)
                ps_nz_audio = self.aug.noise(imd_pss_audio)
                pitch_shift_stretch_noise_feat = self.feat.extract_features(ps_nz_audio)
                # logger.info("Adding AGWN to pitch-shift-stretched audio")
            else:
                logger.error("No transformation parameters specified!")

        audio = np.vstack(
            # tuple([eval(f"{param}_feat") for param in self.tfx_params]),
            (default_feat, noise_feat, pitch_feat, pitch_noise_feat),
            casting="same_kind",
        )
        return audio

    def process_feature(self, path, emotion):
        features = self.get_features(path)
        X = []
        Y = []
        for ele in features:
            X.append(ele)
            Y.append(emotion)
        return X, Y

    def feature_engineering(self):
        root_dir = self.config.root_dir
        metadata_dir = self.config.metadata_path

        data = pd.read_csv(metadata_dir)
        data = data.dropna()
        paths = data["FilePath"]
        emotions = data["Emotions"]
        logger.warning(" !!!! OVERHEAT ALERT !!!! ")
        logger.warning(" ==== Using Multiprocessors for Data Transformation ====")
        num_cpus = mp.cpu_count() - 1
        logger.info(f"Number of processors: {str(num_cpus)}")
        start = timeit.default_timer()
        logger.info(f"Multiprocessing started!")
        # Run the loop in parallel
        results = Parallel(n_jobs=-2)(
            delayed(self.process_feature)(path, emotion)
            for (path, emotion) in zip(paths, emotions)
        )

        # Collect the results
        X = []
        Y = []
        for result in results:
            x, y = result
            X.extend(x)
            Y.extend(y)
        elapsed_time = timeit.default_timer() - start
        logger.info(f"Elapsed Time: {elapsed_time:.2f} secs")

        # Saving the unstructured audio signals into a data frame
        emotions_df = pd.DataFrame(X)
        emotions_df["Emotions"] = Y
        logger.info("Trying to export dataset to disk....")
        start = timeit.default_timer()
        # emotions_df.to_csv(os.path.join(root_dir, "emotion.csv"), index=False)
        emotions_df.to_parquet(self.config.output_path, compression="gzip")
        logger.info("Dataframe written to disk!!")
        elapsed_time = timeit.default_timer() - start
        logger.info(f"Elapsed Time: {elapsed_time:.2f} secs")
        emotions_df = pd.read_parquet(self.config.output_path)
        logger.info(f"Shape of saved Dataframe {str(emotions_df.shape)}")
        logger.info(
            f"Data Types: \n {str(emotions_df.info())}",
        )
        logger.info(f"Descriptive Stats: \n{str(emotions_df.describe(include='all'))}")
        logger.info(
            f"Total Null Values Before Zero Imputation: {str(emotions_df.isna().sum().sum())}"
        )
        # Since the audio signals are of different length fill missing with 0
        emotions_df = emotions_df.fillna(0)
        logger.info(
            f"Total Null Values After Zero Imputation: {str(emotions_df.isna().sum().sum())}"
        )

    def train_test_split_data(self, test_size=0.2):
        emotions_df = pd.read_parquet(self.config.output_path)
        train, test = train_test_split(
            emotions_df, test_size=test_size, random_state=42
        )
        logger.info("Data split into train and test")
        logger.info(f"Train data shape: {str(train.shape)}")
        logger.info(f"Test data shape: {str(test.shape)}")
        train.to_parquet(self.config.train_path, compression="gzip")
        test.to_parquet(self.config.test_path, compression="gzip")
        logger.info("Dataframe written to disk!!")
