import os
import re
import pandas as pd
from mlcore import logger
from mlcore.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def generate_metadata(self) -> bool:
        try:
            output_dir = self.config.local_output_path

            ### =================================================================
            ### Process RAVDESS dataset
            ### =================================================================
            data_dir = self.config.unzip_ravdess_dir
            directory_list = os.listdir(data_dir)
            file_emotion = []
            file_emotion_intensity = []
            emotion_actor_number = []
            file_path = []
            for folder in directory_list:
                if folder.startswith("Actor_"):
                    actor_path = os.path.join(data_dir, folder)
                    actor_files = os.listdir(actor_path)
                    for file in actor_files:
                        part = file.split(".")[0].split("-")
                        file_emotion.append(int(part[2]))
                        file_emotion_intensity.append(int(part[3]))
                        emotion_actor_number.append(int(part[6]))
                        file_path.append(os.path.join(actor_path, file))

            emotion_df = pd.DataFrame(file_emotion, columns=["Emotions"])
            emotion_intns_df = pd.DataFrame(
                file_emotion_intensity, columns=["EmotionIntensity"]
            )
            emotion_actor_df = pd.DataFrame(
                emotion_actor_number, columns=["ActorNumber"]
            )
            path_df = pd.DataFrame(file_path, columns=["FilePath"])
            ravdess_df = pd.concat(
                [emotion_df, emotion_intns_df, emotion_actor_df, path_df], axis=1
            )
            ravdess_df["Dataset"] = "RAVDESS"
            ravdess_df["Emotions"] = ravdess_df["Emotions"].replace(
                {
                    1: "neutral",
                    2: "neutral",
                    3: "happy",
                    4: "sad",
                    5: "angry",
                    6: "fear",
                    7: "disgust",
                    8: "surprise",
                }
            )
            ravdess_df["EmotionIntensity"] = ravdess_df["EmotionIntensity"].replace(
                {
                    1: "MD",
                    2: "LO",
                }
            )

            logger.info(f"Shape of Ravdess dataframe: {ravdess_df.shape[0]}")
            logger.info(ravdess_df.head())

            ### =================================================================
            ### Process TESS dataset
            ### =================================================================

            data_dir = self.config.unzip_tess_dir
            directory_list = os.listdir(data_dir)
            file_emotion = []
            file_emotion_intensity = []
            emotion_actor_number = []
            file_path = []
            for folder in directory_list:
                if folder.startswith("YAF_") or folder.startswith("OAF_"):
                    actor_path = os.path.join(data_dir, folder)
                    actor_files = os.listdir(actor_path)
                    for file in actor_files:
                        part = file.split(".")[0].split("_")
                        file_emotion.append(
                            part[2].lower() if part[2] != "ps" else "surprise"
                        )
                        file_emotion_intensity.append("XX")  # not provided
                        emotion_actor_number.append(1 if part[0] == "YAF" else 2)
                        file_path.append(os.path.join(actor_path, file))

            emotion_df = pd.DataFrame(file_emotion, columns=["Emotions"])
            emotion_intns_df = pd.DataFrame(
                file_emotion_intensity, columns=["EmotionIntensity"]
            )
            emotion_actor_df = pd.DataFrame(
                emotion_actor_number, columns=["ActorNumber"]
            )
            path_df = pd.DataFrame(file_path, columns=["FilePath"])
            tess_df = pd.concat(
                [emotion_df, emotion_intns_df, emotion_actor_df, path_df], axis=1
            )
            tess_df["Dataset"] = "TESS"

            logger.info(f"Shape of TESS dataframe: {tess_df.shape[0]}")
            logger.info(tess_df.head())

            ### =================================================================
            ### Process Crema D dataset
            ### =================================================================

            data_dir = self.config.unzip_cremad_dir
            directory_list = os.listdir(data_dir)
            file_emotion = []
            file_emotion_intensity = []
            emotion_actor_number = []
            file_path = []
            for file in directory_list:
                part = file.split(".")[0].split("_")
                file_emotion.append(part[2])
                file_emotion_intensity.append(part[3])
                emotion_actor_number.append(int(part[0]))
                file_path.append(os.path.join(data_dir, file))

            emotion_df = pd.DataFrame(file_emotion, columns=["Emotions"])
            emotion_intns_df = pd.DataFrame(
                file_emotion_intensity, columns=["EmotionIntensity"]
            )
            emotion_actor_df = pd.DataFrame(
                emotion_actor_number, columns=["ActorNumber"]
            )
            path_df = pd.DataFrame(file_path, columns=["FilePath"])
            cremad_df = pd.concat(
                [emotion_df, emotion_intns_df, emotion_actor_df, path_df], axis=1
            )
            cremad_df["Dataset"] = "CREMAD"
            cremad_df["Emotions"] = cremad_df["Emotions"].replace(
                {
                    "NEU": "neutral",
                    "HAP": "happy",
                    "SAD": "sad",
                    "ANG": "angry",
                    "FEA": "fear",
                    "DIS": "disgust",
                }
            )

            logger.info(f"Shape of CREMAD dataframe: {cremad_df.shape[0]}")
            logger.info(cremad_df.head())

            ### =================================================================
            ### Process SAVEE dataset
            ### =================================================================

            data_dir = self.config.unzip_savee_dir
            directory_list = os.listdir(data_dir)
            file_emotion = []
            file_emotion_intensity = []
            emotion_actor_number = []
            file_path = []
            for file in directory_list:
                part = file.split(".")[0].split("_")
                file_emotion.append(re.search(r"[a-zA-Z]+", part[1]).group())
                file_emotion_intensity.append("XX")  ## not provided
                emotion_actor_number.append(part[0])
                file_path.append(os.path.join(data_dir, file))

            emotion_df = pd.DataFrame(file_emotion, columns=["Emotions"])
            emotion_intns_df = pd.DataFrame(
                file_emotion_intensity, columns=["EmotionIntensity"]
            )
            emotion_actor_df = pd.DataFrame(
                emotion_actor_number, columns=["ActorNumber"]
            )
            path_df = pd.DataFrame(file_path, columns=["FilePath"])
            savee_df = pd.concat(
                [emotion_df, emotion_intns_df, emotion_actor_df, path_df], axis=1
            )
            savee_df["Dataset"] = "SAVEE"
            savee_df["Emotions"] = savee_df["Emotions"].replace(
                {
                    "n": "neutral",
                    "h": "happy",
                    "sa": "sad",
                    "a": "angry",
                    "f": "fear",
                    "d": "disgust",
                    "su": "surprise",
                }
            )
            savee_df["ActorNumber"] = savee_df["ActorNumber"].replace(
                {
                    "DC": "1",
                    "JE": "2",
                    "JK": "3",
                    "KL": "4",
                }
            )

            logger.info(f"Shape of CREMAD dataframe: {savee_df.shape[0]}")
            logger.info(savee_df.head())

            ### =================================================================
            ### Concatenate all datasets
            ### =================================================================

            combined_df = pd.concat([ravdess_df, tess_df, cremad_df, savee_df], axis=0)
            combined_df.to_csv(output_dir, index=False)
            logger.info(f"Shape of Final dataframe: {combined_df.shape}")

            # Validate columns against metadata schema
            validation_status = self._validate_columns(combined_df)

            return validation_status

        except Exception as e:
            logger.error("Data Validation Failed")
            raise e

    def _validate_columns(self, data_df: pd.DataFrame) -> bool:
        try:
            validation_status = None
            data_cols = list(data_df.columns)
            metadata_schema = self.config.metadata_schema.keys()

            for col in data_cols:
                if col not in metadata_schema:
                    validation_status = False
                else:
                    validation_status = True

            with open(self.config.validation_status, "w") as f:
                f.write(f"validation_status: {validation_status}")

            return validation_status

        except Exception as e:
            logger.error("Column validation failed.")
            raise e
