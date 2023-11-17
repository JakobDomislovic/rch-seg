# system imports
import yaml
import datetime
import numpy as np
import pytz

class ConfigLoader:
    """
    Class that loads config with params for datasets, 
    neural network hyperparams, and paths for model saving.
    """
    def __init__(self, path:str=None) -> None:

        with open(path, "r") as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        # job type
        self.job_type = self.config['job_type']
        # organ
        self.organ = self.config['organ']
        # camera view
        self.camera_view = self.config['view']
        # weigths and biases
        self.use_wandb = self.config['use_wandb']
        # timezone
        self.timezone = pytz.timezone('Europe/Zagreb')
        # timestamp that will be used for filenames, etc.
        self.timestamp = datetime.datetime.now(tz=self.timezone).strftime(self.config['save_paths']['time_and_date_format'])
        
    def load_train_config(self):
        # data
        self.data_paths = self.config['data']['paths']
        self.train_val_test_ratio = []
        for ratio in self.config['data']['train_validation_test_split_ratio']:
            d = {
                "seed": ratio[0],
                "train": ratio[1],
                "validation": ratio[2],
                "test": ratio[3], 
            }
            self.train_val_test_ratio.append(d)
        self.remove_dirty_data = self.config['data']['dirty_data']['remove_dirty_data']
        self.dirty_data_paths = self.config['data']['dirty_data']['paths']
        self.images_per_patient = self.config['data']['images_per_patient']
        self.augmentation = self.config['data']['augmentation']
        # load backbone
        self.backbone_path = self.config['backbone']['backbone_path']
        self.backbone = self.config['backbone']['name']
        # organ merging
        self.using_organ_merging = self.config['organ_merging']['using_organ_merging']
        self.number_of_merged_organs = self.config['organ_merging']['number_of_merged_organs']
        # save paths
        self.model_path = self.config['save_paths']['model']
        self.patients_path = self.config['save_paths']['patients']
        self.save_weights = self.config['save_paths']['save_weights']
        # load model config
        self.load_model_config()
    
    def load_retrain_config(self):
        # data
        self.data_paths = self.config['data']['paths']
        self.train_val_test_ratio = []
        for ratio in self.config['data']['train_validation_test_split_ratio']:
            d = {
                "seed": ratio[0],
                "train": ratio[1],
                "validation": ratio[2],
                "test": ratio[3], 
            }
            self.train_val_test_ratio.append(d)
        self.remove_dirty_data = self.config['data']['dirty_data']['remove_dirty_data']
        self.dirty_data_paths = self.config['data']['dirty_data']['paths']
        self.images_per_patient = self.config['data']['images_per_patient']
        self.augmentation = self.config['data']['augmentation']
        # timestamp that will be used for filenames, etc.
        self.timestamp = datetime.datetime.now(tz=self.timezone).strftime(self.config['save_paths']['time_and_date_format'])
        # load pretrained model
        self.trained_model_path = self.config['trained_model']['path']
        self.trained_model_name = self.config['trained_model']['name']
        # organ merging
        self.using_organ_merging = self.config['organ_merging']['using_organ_merging']
        self.number_of_merged_organs = self.config['organ_merging']['number_of_merged_organs']
        # patients distribution from training
        self.using_patient_distribution = self.config['patient_distribution_from_training']['using']
        if self.using_patient_distribution == True:
            self.remove_dirty_data = False
        self.patients_path = self.config['patient_distribution_from_training']['path']
        # save paths
        self.time_and_date_format = self.config['save_paths']['time_and_date_format']
        self.model_path = self.config['save_paths']['model']
        self.save_weights = self.config['save_paths']['save_weights']
        # load model config
        self.load_model_config()
    
        
    def load_evaluation_config(self):
        # data
        self.evaluation_data_paths = self.config['data']['paths']
        self.remove_dirty_data = self.config['data']['dirty_data']['remove_dirty_data']
        self.dirty_data_paths = self.config['data']['dirty_data']['paths']
        self.using_image_range = self.config['data']['specify_image_range']['using']
        self.image_range = self.config['data']['specify_image_range']['ranges']
        self.images_per_patient = self.config['data']['images_per_patient']
        # load pretrained model
        self.trained_model_path = self.config['trained_model']['path']
        self.trained_model_name = self.config['trained_model']['name']
        # patients distribution from training
        self.using_patient_distribution = self.config['patient_distribution_from_training']['using']
        self.patients_path = self.config['patient_distribution_from_training']['path']
        # threshold while predicting
        self.threshold = self.config['score_threshold']
        # organ merging
        self.using_organ_merging = self.config['organ_merging']['using_organ_merging']
        self.number_of_merged_organs = self.config['organ_merging']['number_of_merged_organs']
        # export images with score below threshold
        self.export_score = self.config['export_score']['using']
        self.score_threshold = self.config['export_score']['threshold']
        self.export_score_path = self.config['export_score']['path']
        # load model config
        self.load_model_config()
    
    def load_prediction_config(self):
        # data
        self.evaluation_data_paths = self.config['data']['paths']
        self.using_image_range = self.config['data']['specify_image_range']['using']
        self.image_range = self.config['data']['specify_image_range']['ranges']
        self.images_per_patient = self.config['data']['images_per_patient']
        # load pretrained model
        self.trained_model_path = self.config['trained_model']['path']
        self.trained_model_name = self.config['trained_model']['name']
        # patients distribution from training
        self.patient_distribution_from_training = self.config['patient_distribution_from_training']['path']
        # threshold while predicting
        self.threshold = self.config['score_threshold']
        # organ merging
        self.using_organ_merging = self.config['organ_merging']['using_organ_merging']
        self.number_of_merged_organs = self.config['organ_merging']['number_of_merged_organs']
        # save path
        self.save_prediction_path = self.config['save_path']
        # load model config
        self.load_model_config()
    
    def load_model_config(self):
       # segmentation model
       self.compile = self.config['segmentation_model']['compile']
       self.batch_size = self.config['segmentation_model']['batch_size']
       self.number_of_epochs = self.config['segmentation_model']['epochs']
       self.backbone = self.config['segmentation_model']['backbone']
       self.loss = self.config['segmentation_model']['loss']
       self.activation = self.config['segmentation_model']['activation']
       self.number_of_classes = self.config['segmentation_model']['number_of_classes']
       self.encoder_weights = self.config['segmentation_model']['encoder_weights']
       self.encoder_freeze = self.config['segmentation_model']['encoder_freeze']
       # segmentation model -- scheduler
       # exponential
       self.using_exponential = self.config['segmentation_model']['scheduler']['exponential']['using']
       self.initial_learning_rate_exp = np.float32(self.config['segmentation_model']['scheduler']['exponential']['initial_learning_rate'])
       self.decay_steps = self.config['segmentation_model']['scheduler']['exponential']['decay_steps']
       self.decay_rate = np.float32(self.config['segmentation_model']['scheduler']['exponential']['decay_rate'])
       # cosine
       self.using_cosine = self.config['segmentation_model']['scheduler']['cosine']['using']
       self.initial_learning_rate_cos = self.config['segmentation_model']['scheduler']['cosine']['initial_learning_rate']
       self.first_decay_steps = self.config['segmentation_model']['scheduler']['cosine']['first_decay_steps']
       self.t_mul = self.config['segmentation_model']['scheduler']['cosine']['t_mul']
       self.m_mul = self.config['segmentation_model']['scheduler']['cosine']['m_mul']
       self.alpha = self.config['segmentation_model']['scheduler']['cosine']['alpha']
       # callback
       self.early_stopping_monitor = self.config['segmentation_model']['callbacks']['early_stopping']['early_stopping_monitor']
       self.early_stopping_mode = self.config['segmentation_model']['callbacks']['early_stopping']['early_stopping_mode']
       self.early_stopping_patience = self.config['segmentation_model']['callbacks']['early_stopping']['early_stopping_patience']
       self.early_stopping_restore_best = self.config['segmentation_model']['callbacks']['early_stopping']['early_stopping_restore_best']
       self.early_stopping_baseline = self.config['segmentation_model']['callbacks']['early_stopping']['early_stopping_baseline']
       self.min_delta = self.config['segmentation_model']['callbacks']['early_stopping']['early_stopping_min_delta']
