"""
File: finetune.py
Author: Yuk-Hoi Yiu
Description: 
    This is a minimal example script to show how to fine-tune or retrain the model, based on procedures detailed in Yiu et al., 2019.
    The actual training script used in the paper was unfortunately lost.
    MLflow tracking and logging are included. Validation, hyperparameter search, testing and storing checkpoints are not covered in this script.
Usage:
    See fine_tune(...).
"""
from deepvog.model.DeepVOG_model import DeepVOG_net
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import IoU

import mlflow

class MLflowLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric, value in logs.items():
            mlflow.log_metric(metric, value, step=epoch)

def create_generator(image_datagen, mask_datagen, data_dir, batch_size) -> tuple:
    input_generator = image_datagen.flow_from_directory(
        data_dir,
        target_size=(240, 320),
        color_mode='rgb',
        class_mode=None,
        classes = ['images'],
        batch_size=batch_size,
        seed=1
    )
    
    label_generator = mask_datagen.flow_from_directory(
        data_dir,
        target_size=(240, 320),
        color_mode='rgb',
        class_mode=None,
        classes = ['masks'],
        batch_size=batch_size,
        seed=1
    )
    
    # return train_generator
    while True:
        input_images = next(input_generator)
        label_images = next(label_generator)
        yield (input_images, label_images)


def fine_tune(model_weights: str, data_dir:str, save_model_weights: str, batch_size:int = 32, epochs:int = 10, lr:float=0.001,
              mlflow_tracking_uri: str | None = None, mlflow_log_model: bool = False) -> None:
    """Fine tune the DeepVOG model weights to your data with optional MLflow tracking.

    Parameters
    ----------
    model_weights : str
        Path to the .h5 file of the model weights.
    data_dir : str
        Directory where you store the training data. The Directory should be structured as follows:
        - data_dir
            - images
            - masks
        data_dir/images should contain (240, 320, 3) .png/.jpg images with rgb channels.
        data_dir/masks should contain (240, 320, 3) .png/.jpg images with rgb channels. where all values are 0, except the second channel mask[:, :, 1] has
        values 255 segmented areas (converted to 1.0 as the label in keras). The names must correspond to the images in data_dir/images
    
    save_model_weights : str
        Path to save the model weights. The filename should end with ".weights.h5"
    
    batch_size : int, optional
    epochs : int, optional
    lr : float, optional
    
    mlflow_tracking_uri : str or None, optional
        Disable MLflow tracking by setting it to None. 
    
    mlflow_log_model : bool, optional
    """
    
    # Set up mlflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment('Fine-tuning DeepVOG')
        mlflow.start_run()
        callbacks = [MLflowLogger()]
    else:
        callbacks = None

    # Load model
    model = DeepVOG_net()
    model.load_weights(model_weights)

    # Data Generator with Augmentation
    data_gen_args = dict(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        rescale = 1/255,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    train_generator = create_generator(image_datagen, mask_datagen, data_dir, batch_size)

    # Compile the model
    adam = Adam(learning_rate=lr)
    loss = CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="categorical_crossentropy",
    )
    metrics = IoU(num_classes=2, target_class_ids=[0])

    # One could use Dice's coefficient as loss too. 
    # During our training, we used Dice's coefficient as metric (not as loss) for evaluation.
    model.compile(optimizer=adam, loss=loss, metrics=[metrics])

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=epochs // batch_size,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save the model
    if save_model_weights:
        model.save_weights(save_model_weights)

    if mlflow.active_run():
        if mlflow_log_model:
            mlflow.keras.log_model(model, "model")
        mlflow.end_run()

if __name__ == '__main__':

    fine_tune(model_weights='deepvog/model/DeepVOG_weights.h5',
              data_dir='data',
              save_model_weights='DeepVOG_finetuned.weights.h5',
              batch_size=2,
              epochs=5, 
              lr = 0.0001,
              mlflow_tracking_uri = None,  # Disable mlflow tracking. Enter your own MLflow tracking uri.
              mlflow_log_model = False)
            