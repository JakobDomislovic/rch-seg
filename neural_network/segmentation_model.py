# system imports
from typing import List
import numpy as np
import pandas as pd
from math import log, exp, cosh
# self imports
from loaders.config_loader import ConfigLoader
from neural_network.loss_functions import dice_loss, bce_dice_loss, balanced_binary_crossentropy, weighted_binary_crossentropy, tversky_loss, focal_tversky, custom_loss, log_cosh_jaccard
# segmentation_models library
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
# tensorflow and keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History
import tensorflow.keras.backend as K
from keras.callbacks import LambdaCallback
from keras.losses import binary_crossentropy, binary_focal_crossentropy, log_cosh, hinge
# weights and biases
import wandb
from wandb.keras import WandbCallback

class SegmentationModel:
    """
    Class that creates or loads segmentation model. 
    """
    def __init__(self, config:ConfigLoader=None) -> None:
        self.config = config
        if self.config.use_wandb:
            self.run_wandb = wandb.init(
                project="rch-loss", #"radiochirurgia-segmentation", 
                job_type=self.config.job_type + " " + self.config.organ,
                #group=self.config.organ,
                name=self.config.job_type + " " + self.config.organ + " " + self.config.timestamp,
                config=self.config, 
                entity="jdomislovic")
            #self.setup_wandb()
        self.beta = K.variable(value=0)
        self.wu_cb = LambdaCallback(on_epoch_end=lambda epoch, log: self.warmup(epoch))
        # define scheduler
        self.scheduler = self.define_scheduler()
        # define optimizer
        self.optimizer = self.define_optimizer()
        # define callbacks
        self.callbacks = self.define_callbacks()
    
    # define callback to change the value of beta at each epoch
    def warmup(self, epoch):
        value = epoch
        print("beta:", value)
        K.set_value(self.beta, value)

    def create_model(self) -> None:
        preprocess_input = sm.get_preprocessing(self.config.backbone)
        # define segmentation model
        base_model = sm.Unet(
            backbone_name=self.config.backbone,
            classes = 1,
            activation=self.config.activation,
            encoder_weights=self.config.encoder_weights,
            encoder_freeze=self.config.encoder_freeze,
        )
        if self.config.using_organ_merging:
            input = Input(shape=(224, 224, 1+self.config.number_of_merged_organs), name='MergedInput')
            convolution_layer = SeparableConv2D(3, (1, 1), name='SeparableConv2D')(input) # map N channels to 3 channels
            preprocessed = preprocess_input(convolution_layer)
        else:
            input = Input(shape=(224, 224, 3), name='Input')
            preprocessed = preprocess_input(input)
        output = base_model(preprocessed)
        # final model
        model = Model(input, output, name='BaseModel')
        
        self.model = model

    def compile(self) -> None:
        tf.keras.optimizers.Adam(
            learning_rate= self.scheduler,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name='Adam',
        )
        tf.keras.losses.BinaryFocalCrossentropy(
            gamma=2.0, # parametar koliko smanjujemo loss dobrih klasifikacija (1-s)^gamma * g*log(s) gdje je s segmentacija, a ground truth
            name='binary_focal_crossentropy'
        )

        if self.config.loss == 'Jaccard':
                loss = sm.losses.bce_jaccard_loss
        elif self.config.loss == 'Dice':
            loss = self.dice_coef_loss, #sm.losses.dice_loss
        elif self.config.loss == 'Focal':
            loss = sm.losses.BinaryFocalLoss(alpha=0.8, gamma=2.0)
        elif self.config.loss == 'bce_dice':
            loss = bce_dice_loss
        elif self.config.loss == 'balanced_bce':
            loss = balanced_binary_crossentropy 
        elif self.config.loss == 'weighted_bce':
            loss = weighted_binary_crossentropy
        elif self.config.loss == 'Tversky':
            loss = tversky_loss
        elif self.config.loss == 'focal_Tversky':
            loss = focal_tversky
        elif self.config.loss == 'galdran':
            loss = self.galdran_bce_dice_loss
        elif self.config.loss == 'logcosh':
            loss = log_cosh
        elif self.config.loss == 'custom':
            loss = custom_loss
        elif self.config.loss == 'log_cosh_jaccard':
            loss = log_cosh_jaccard
        
        self.model.compile(
            optimizer='Adam',
            loss=loss,
            metrics=['binary_accuracy', sm.metrics.iou_score, sm.metrics.f1_score]
        )

    def fit(self, train_data, validation_data) -> History:
        if self.config.use_wandb:
            self.callbacks.append(WandbCallback())
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=self.config.number_of_epochs,
            callbacks=self.callbacks,
        )
        return history
    
    def predict(self, img, threshold=0.5) -> None:
        prediction = self.model.predict(img) > threshold
        return prediction

    def evaluate(self, eval_data) -> None:
        evaluation = self.model.evaluate(
            eval_data,
            verbose=1,
            return_dict=True,
        )
        if self.config.use_wandb:
            self.run_wandb.log({
                "test_loss": evaluation["loss"],
                "test_acc": evaluation["binary_accuracy"],
                "test_iou": evaluation["iou_score"],
                "test_f1": evaluation["f1-score"],
            })

    def save_model(self) -> None:
        save_path = self.config.model_path + self.config.job_type + '_' + self.config.organ + '_' + self.config.timestamp
        self.model.save(
            save_path,
            save_format="tf",
            include_optimizer=True,
        )

    def load_model(self) -> None:
        load_path = self.config.trained_model_path + self.config.trained_model_name
        model = tf.keras.models.load_model(
            load_path, 
            custom_objects={"iou_score": sm.metrics.iou_score, "f1_score": sm.metrics.f1_score},
            compile=False,
        )
        self.model =  model
    
    def define_scheduler(self) -> tf.keras.optimizers.schedules:
        lr_scheduler_exp = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.initial_learning_rate_exp,
            decay_steps=self.config.decay_steps,
            decay_rate=self.config.decay_rate,
        )
        lr_scheduler_cos = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=self.config.initial_learning_rate_cos,
            first_decay_steps=self.config.first_decay_steps,
            t_mul=self.config.t_mul,
            m_mul=self.config.m_mul,
            alpha=self.config.alpha,
        )
        if self.config.using_exponential:
            return lr_scheduler_exp
        else:
            return lr_scheduler_cos

    def define_optimizer(self) -> tf.keras.optimizers:
        adam = tf.keras.optimizers.Adam(
            learning_rate= self.scheduler,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name='Adam',
        )
        return adam

    def define_callbacks(self) -> List:
        lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(
            self.scheduler, 
            verbose=1,
        )
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor=self.config.early_stopping_monitor,
            min_delta=self.config.min_delta,
            patience=self.config.early_stopping_patience,
            verbose=1,
            mode=self.config.early_stopping_mode,
            baseline=self.config.early_stopping_baseline,
            restore_best_weights=self.config.early_stopping_restore_best,
        )
        
        callbacks = [lr_scheduler_callback, early_stopping_callback, self.wu_cb]
        return callbacks

    def precision_recall_curve(self, data):
        # calculate precision-recall curve data and put in 
        df_threshold = np.array([])
        df_precision = np.array([])
        df_recall = np.array([])
        for img, msk in data.as_numpy_iterator():
            prec_rec = dict()
            threshold_list = []
            precision_list = []
            recall_list = []
            for i in np.arrange(0.5, 1.00, 0.02):
                threshold = round(i, 2)
                pred = self.model.predict(img) > i
                pred = pred.flatten()
                mask = msk.flatten()
                tp, tn, fp, fn = self.calculate_tp_tn_fp_fn(pred, mask)
                precision = (tp) / (tp + fp)
                recall = (tp) / (tp + fn)

                threshold_list.append(threshold)
                precision_list.append(precision)
                recall_list.append(recall)

            df_threshold = np.stack(df_threshold, threshold_list)
            df_precision = np.stack(df_precision, precision_list)
            df_recall = np.stack(df_recall, recall_list)
        
        df = pd.DataFrame(columns=['Threshold', 'Precision', 'Recall'])
        df['Threshold'] = np.mean(df_threshold, axis=0)
        df['Precision'] = np.mean(df_precision, axis=0)
        df['Recall'] = np.mean(df_recall, axis=0)

        return df
    
    def dice_coef(self, y_true, y_pred, smooth=1e-5):        
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return dice
    
    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def calculate_tp_tn_fp_fn(self, pred, mask):
        # calculate true positive (tp), true negative (tn), false positive (fp), false negative (fn)
        tp = ((pred == 1) & (mask == 1)).sum()
        tn = ((pred == 0) & (mask == 0)).sum()
        fp = ((pred == 1) & (mask == 0)).sum()
        fn = ((pred == 0) & (mask == 1)).sum()

        return tp, tn, fp, fn
    

    ### LOSS
    def galdran_bce_dice_loss(self, y_true, y_pred):
        loss = ((50-self.beta)/50)*binary_crossentropy(y_true, y_pred) + (self.beta/50)*dice_loss(y_true, y_pred)
        return loss