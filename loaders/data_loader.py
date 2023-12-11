# system imports
import numpy as np
import os 
from sklearn.model_selection import train_test_split
from itertools import permutations
import pandas as pd
# self imports
from loaders.config_loader import ConfigLoader
# tensorflow
import tensorflow as tf
import tf_clahe


class DataLoader:
    """
    Class for dataloading with help of tensorflow.
    """
    def __init__(self, config:ConfigLoader=None) -> None:
        self.config = config
        self.images = "Images/"
        self.masks = "Masks" + self.config.organ + "/"
    
    def load_image_and_mask(self, image, mask, num_classes=1, size=(224,224)):
        # load image
        print(image)
        img = tf.io.read_file(image)
        img = tf.image.decode_png(contents=img, channels=3)
        # CLAHE
        img = tf.cast(x=img, dtype=tf.float32)
        if self.config.using_organ_merging:
            img = img[:, :, 0:(1+self.config.number_of_merged_organs)]
        # load mask
        mask = tf.io.read_file(mask)
        mask = tf.image.decode_png(contents=mask, channels=1)
        # CLAHE
        mask = tf.cast(x=mask, dtype=tf.float32)

        img = tf.image.random_crop(value=img, size=(96, 96, 3), seed=1)
        mask = tf.image.random_crop(value=mask, size=(96, 96, 1), seed=1)
        return img, mask

    def dataset_pipeline(self, images, masks, batch_size=1, num_classes=1, size=(224,224)):
        # load datasets
        dataset = tf.data.Dataset.from_tensor_slices((images, masks))
        dataset = dataset.shuffle(buffer_size=200)
        dataset = dataset.map(lambda x,y: self.load_image_and_mask(x, y, num_classes, size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
        return dataset
    
    def load_train_and_retrain_dataset(self):
        
        # this is for saving patients in .csv
        patient_subset = []
        patient_number = []
        patient_mask_number = []
        patients_dataset = []
        
        # extract dirty data if using that feauture
        if not self.config.remove_dirty_data:
            dirty_data = [None for i in range(len(self.config.data_paths))]
        elif self.config.remove_dirty_data:
            # create list of lists with dirty data for each dataset
            dirty_data = self.extract_dirty_data()
        # initialize data dict
        subsets = ("train", "validation", "test")
        k = ("Patients", "Data")
        x = {key: {i: []  for i in subsets}  for key in k}
        y = {key: {i: []  for i in subsets}  for key in k}
        datasets = dict.fromkeys(subsets, [])
        loop_params = zip(self.config.data_paths, self.config.train_val_test_ratio, dirty_data)
        for dataset_path, train_val_test_ratio, dirty_data in loop_params:
            # reset patients in x and y, but don't reset data
            for key in x["Patients"].keys():
                x["Patients"][key] = []
                y["Patients"][key] = []
            # first we load images and masks paths
            images = sorted(os.listdir(dataset_path + self.images))
            masks = sorted(os.listdir(dataset_path + self.masks))
            images_list = sorted([os.path.join(dataset_path + self.images, img_name) for img_name in images])
            masks_list = sorted([os.path.join(dataset_path + self.masks, mask_name) for mask_name in masks])
            masks_set = set(masks)
            # all patients images (e.g. 1200.png-1799.png) that have mask inside mask folder
            images_with_masks = [i for i, e in enumerate(images) if e in masks_set]
            # here are patient numbers for which we have mask (e.g. 2,3,5,6 if patient 1 and 4 don't have mask)
            patients_with_masks = [i // self.config.images_per_patient for i in images_with_masks if (i % self.config.images_per_patient == 0)]
            # here are indicies of mask in mask folder for patients_with_mask (patient 2 has mask but his mask start on index 0 in mask folder)
            patients_mask_index = range(len(patients_with_masks))
            
            # we create train, validation and test subsets based on PATIENTS
            seed = train_val_test_ratio["seed"]
            train_ratio = train_val_test_ratio["train"]
            validation_ratio = train_val_test_ratio["validation"]
            test_ratio = train_val_test_ratio["test"]
            if train_ratio == 1.0:
                x["Patients"]["train"] = patients_with_masks
                y["Patients"]["train"] = patients_mask_index
            elif validation_ratio == 1.0:
                x["Patients"]["validation"] = patients_with_masks
                y["Patients"]["validation"] = patients_mask_index
            elif test_ratio == 1.0:
                x["Patients"]["test"] = patients_with_masks
                y["Patients"]["test"] = patients_mask_index
            elif train_ratio == 0.0:
                x["Patients"]["validation"], x["Patients"]["test"], y["Patients"]["validation"], y["Patients"]["test"] = train_test_split(patients_with_masks, patients_mask_index, test_size = test_ratio, random_state=seed)
            elif validation_ratio == 0.0:
                x["Patients"]["train"], x["Patients"]["test"], y["Patients"]["train"], y["Patients"]["test"] = train_test_split(patients_with_masks, patients_mask_index, test_size = test_ratio, random_state=seed)
            elif test_ratio == 0.0:
                x["Patients"]["train"], x["Patients"]["validation"], y["Patients"]["train"], y["Patients"]["validation"] = train_test_split(patients_with_masks, patients_mask_index, test_size = validation_ratio, random_state=seed)
            else:
                # first we split to train and test and leave test as it is
                x["Patients"]["train"], x["Patients"]["test"], y["Patients"]["train"], y["Patients"]["test"] = train_test_split(patients_with_masks, patients_mask_index, test_size=test_ratio, random_state=seed)
                # we then split train to real train and validation
                x["Patients"]["train"], x["Patients"]["validation"], y["Patients"]["train"], y["Patients"]["validation"] = train_test_split(x["Patients"]["train"], y["Patients"]["train"], test_size=validation_ratio/(train_ratio + validation_ratio), random_state=seed)

            for subset in subsets:
                x_data, y_data = self.create_subsets(
                    x["Patients"][subset], y["Patients"][subset],
                    images_list=images_list, masks_list=masks_list,
                    dirty_data=dirty_data,
                )
                x["Data"][subset] += (x_data)
                y["Data"][subset] += (y_data)
                
            # this is for patient distribution .csv
            for patient, patient_indicies in zip(patients_with_masks, patients_mask_index):
                patient_number.append(patient)
                patient_mask_number.append(patient_indicies)
                for subset in subsets:
                    if patient in x["Patients"][subset]:
                        patient_subset.append(subset)
                        break
                patients_dataset.append(dataset_path)
        
        for subset in subsets:
            datasets[subset] = self.dataset_pipeline(
                images=x["Data"][subset],
                masks=y["Data"][subset],
                batch_size=self.config.batch_size,
            )
        
        # save subset patients to .csv
        patient_distribution = {
            "patient": patient_number,
            "mask_number": patient_mask_number,
            "subset": patient_subset,
            "dataset": patients_dataset
        }
        df = pd.DataFrame(patient_distribution)
        csv_name = self.config.patients_path + self.config.job_type + "_" + self.config.organ + "_" + self.config.timestamp + ".csv"
        df.to_csv(csv_name, index=False)

        return datasets, df

    def create_subsets(self, x_patients, y_patients, images_list, masks_list, dirty_data=None):
        x_subset = []
        y_subset = []
        for x, y in zip(x_patients, y_patients):
            images_range = range(x*self.config.images_per_patient, x*self.config.images_per_patient + self.config.images_per_patient)
            masks_range = range(y*self.config.images_per_patient, y*self.config.images_per_patient + self.config.images_per_patient)
            if self.config.remove_dirty_data:
                for i, j in zip(images_range, masks_range):
                    if i not in dirty_data:
                        x_subset.append(images_list[i])
                        y_subset.append(masks_list[j])
            else:
                for i, j in zip(images_range, masks_range):
                    x_subset.append(images_list[i])
                    y_subset.append(masks_list[j])

        return x_subset, y_subset
    
    def extract_dirty_data(self):
        dirty_data = []
        for dirty_data_paths in self.config.dirty_data_paths:
            dirty = []
            # each dataset has multiple dirty subsets (white, foggy, etc.)
            for path in dirty_data_paths:
                df = pd.read_csv(path)
                # for each csv we will find image number based on patient, drr and view
                for i in range(len(df)):
                    image_number = int(df.loc[i, "Patient"]*300 + df.loc[i, "DRR"]*30 + df.loc[i, "View"])
                    # image_number = int(df.loc[i, "Patient"]*2400 + df.loc[i, "DRR"]*4*30 + df.loc[i, "View"]*4 + df.loc[i, "aug"])
                    if image_number not in dirty:
                        dirty.append(image_number)
            dirty_data.append(dirty)
        return dirty_data
    
    def load_evaluation_and_prediction_data(self):
        x_eval = []
        y_eval = []
        # extract dirty data if using that feauture
        if not self.config.remove_dirty_data:
            dirty_data = [None for i in range(len(self.config.evaluation_data_paths))]
        elif self.config.remove_dirty_data:
            # create list of lists with dirty data for each dataset
            dirty_data = self.extract_dirty_data()
        for dataset_path, ranges, dirty_data in zip(self.config.evaluation_data_paths, self.config.image_range, dirty_data):
            # first we load images and masks paths
            images = sorted(os.listdir(dataset_path + self.images))
            masks = sorted(os.listdir(dataset_path + self.masks))
            images_list = sorted([os.path.join(dataset_path + self.images, img_name) for img_name in images])
            masks_list = sorted([os.path.join(dataset_path + self.masks, mask_name) for mask_name in masks])
            masks_set = set(masks)
            if self.config.using_patient_distribution:
                patients_csv = pd.read_csv(self.config.patients_path + self.config.trained_model_name + '.csv')
                patients_with_masks, patients_mask_index = self.extract_patients_from_csv(patients_csv, dataset_path, 'test')
            else:
                # all patients images (e.g. 1200.png-1799.png) that have mask inside mask folder
                images_with_masks = [i for i, e in enumerate(images) if e in masks_set]
                # here are patient numbers for which we have mask (e.g. 2,3,5,6 if patient 1 and 4 don't have mask)
                patients_with_masks = [i // self.config.images_per_patient for i in images_with_masks if (i % self.config.images_per_patient == 0)]
                # here are indicies of mask in mask folder for patients_with_mask (patient 2 has mask but his mask start on index 0 in mask folder)
                patients_mask_index = range(len(patients_with_masks))
            
            x_data, y_data = self.create_subsets(
                    patients_with_masks, patients_mask_index,
                    images_list=images_list, masks_list=masks_list,
                    dirty_data=dirty_data,
                )
            x_eval += x_data
            y_eval += y_data
        
        eval_dataset = self.dataset_pipeline(images=x_eval, masks=y_eval, batch_size=self.config.batch_size)
        datapaths = {"x_train":x_eval, 
                     "y_train":y_eval}
        return {"eval_dataset" : eval_dataset}, datapaths

    def load_data_from_patient_distribution(self):
        # load patient data from csv file
        patients_csv = pd.read_csv(self.config.patients_path + self.config.trained_model_name + '.csv')
        
        # initialize data dict
        subsets = ("train", "validation", "test")
        k = ("Patients", "Data")
        x = {key: {i: []  for i in subsets}  for key in k}
        y = {key: {i: []  for i in subsets}  for key in k}
        datasets = dict.fromkeys(subsets, [])

        for dataset_path in self.config.data_paths:
            
            images = sorted(os.listdir(dataset_path + self.images))
            masks = sorted(os.listdir(dataset_path + self.masks))
            images_list = sorted([os.path.join(dataset_path+ self.images, img_name) for img_name in images])
            masks_list = sorted([os.path.join(dataset_path+ self.masks, mask_name) for mask_name in masks])
            
            for subset in subsets:
                patients, patients_masks = self.extract_patients_from_csv(patients_csv, dataset_path, subset_type=subset)
                x_data, y_data = self.create_subsets(
                    patients, patients_masks,
                    images_list=images_list, masks_list=masks_list
                )
                x["Data"][subset] += (x_data)
                y["Data"][subset] += (y_data)
            
        for subset in subsets:
            datasets[subset] = self.dataset_pipeline(
                images=x["Data"][subset],
                masks=y["Data"][subset],
                batch_size=self.config.batch_size,
            )

        return datasets, patients_csv
    
    def extract_patients_from_csv(self, csv_file, dataset_path, subset_type):
        patients = list(csv_file[(csv_file.dataset == dataset_path) & (csv_file.subset == subset_type)]['patient'])
        patients_masks = list(csv_file[(csv_file.dataset == dataset_path) & (csv_file.subset == subset_type)]['mask_number'])
        return patients, patients_masks