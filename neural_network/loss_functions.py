import numpy as np
from math import log, cosh
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.losses import binary_crossentropy, binary_focal_crossentropy, log_cosh, hinge


def calculate_tp_tn_fp_fn(pred, mask):
    # calculate true positive (tp), true negative (tn), false positive (fp), false negative (fn)
    tp = ((pred == 1) & (mask == 1)).sum()
    tn = ((pred == 0) & (mask == 0)).sum()
    fp = ((pred == 1) & (mask == 0)).sum()
    fn = ((pred == 0) & (mask == 1)).sum()
    return tp, tn, fp, fn


def dice_coef(y_true, y_pred, smooth=1e-5):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def jaccard_coef(y_true, y_pred, smooth=1e-5):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    intersection = true_pos
    union = true_pos + false_neg + false_pos
    return (intersection+smooth) / (union+smooth)

def jaccard_loss(y_true, y_pred):
    return 1 - jaccard_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def tversky(y_true, y_pred, smooth=1e-5, beta=0.9):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    return (true_pos + smooth)/(true_pos + beta*false_neg + (1-beta)*false_pos + smooth)


def tversky_loss(y_true, y_pred):
    # it emphasizes false negatives
    return 1 - tversky(y_true,y_pred)


def focal_tversky(y_true,y_pred, gamma=2):
    pt_1 = tversky(y_true, y_pred)
    return K.pow((1-pt_1), gamma)


def weighted_binary_crossentropy(y_true, y_pred, beta=2):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())  
    term_1 = y_true * K.log(y_pred + K.epsilon())
    return -K.mean(beta*term_0 + term_1, axis=0)


def balanced_binary_crossentropy(y_true, y_pred, beta=0.8):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())  
    term_1 = y_true * K.log(y_pred + K.epsilon())
    return -K.mean(beta*term_0 + (1-beta)*term_1, axis=0)

def custom_loss(y_true, y_pred):
    return 0.9*focal_tversky(y_true, y_pred) + log_cosh(y_true, y_pred)

def log_cosh_jaccard(y_true, y_pred):
    jaccard = jaccard_loss(y_true, y_pred)
    cosh_custom = (K.exp(jaccard) + K.exp(-jaccard)) / 2
    return K.log(cosh_custom)

def confusion(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg) 
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall