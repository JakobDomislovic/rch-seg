# system imports
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

# self imports
from loaders.config_loader import ConfigLoader
from loaders.data_loader import DataLoader
from neural_network.segmentation_model import SegmentationModel

if __name__ == "__main__":
    
    # load config file
    config = ConfigLoader("./config/train_config.yaml")
    config.load_train_config()
    
    # create needed directories
    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(config.patients_path, exist_ok=True)
    
    # load data
    data = DataLoader(config=config)
    subsets, patients_df = data.load_train_and_retrain_dataset()
    train_data = subsets["train"]
    validation_data = subsets["validation"]
    test_data = subsets["test"]

    # create segmentation model
    model = SegmentationModel(config=config)
    if config.use_wandb:
        model.run_wandb.log({"Patients_distribution": patients_df})
    model.create_model()
    model.compile()
    # training the model
    model.fit(train_data=train_data, validation_data=validation_data)
    model.save_model()
    # evaluating the model
    model.evaluate(eval_data=test_data)

