import os
import json
import time
import numpy as np

from _0_utils import print_config
from _0_utils import load_callbacks
from _0_utils import save_training_history
from _0_utils import plot_training_summary
from _2_dataset import load_dataset
from _4_model import build_model

def run():
    # Loading the running configuration
    config = json.load(open("_0_config_remake.json", "r")) 
    # thay đổi thành _0_config_remake.json
    # _0_config_remake.json chỉ thay đổi các giá trị tăng cường, không thay đổi các giá trị làm ảnh hưởng các file python trước (như img_height, img_width, ...)
    # nên các file python trước không cần đổi _0_config.json thành _0_config_remake.json
    # nhưng nếu muốn đồng nhất, thì đổi các file python trước thành _0_config_remake.json cũng được không vấn đề gì
    print_config(config)

    # Loading the dataloaders
    train_generator, valid_generator, test_generator = load_dataset()

    # Loading the model
    model = build_model()

    # Training the model
    start = time.time()

    train_history = model.fit(train_generator,
                              epochs=config["epochs"],
                              steps_per_epoch= len(train_generator) // config["batch_size"],
                              validation_data=valid_generator,
                              validation_steps=len(valid_generator),
                              callbacks=load_callbacks(config))
    end = time.time()

    # Saving the model
    if not os.path.exists(config["checkpoint_filepath"]):
        print(f"[INFO] Creating directory {config['checkpoint_filepath']} to save the trained model")
        os.mkdir(config["checkpoint_filepath"])
    print(f"[INFO] Saving the model and log in \"{config['checkpoint_filepath']}\" directory")
    model.save(os.path.join(config["checkpoint_filepath"], 'saved_model_with_aug.keras'))

    # Process train_history.history. Thay đổi để đưa vào lưu file csv không đổi
    # thêm None vào cột trống, thay đổi learning_rate thành lr để thích hợp với file utils.py
    history = train_history.history
    max_length = max(len(v) for v in history.values())

    for key, value in history.items():
        if len(value) < max_length:
            history[key].extend([None] * (max_length - len(value)))

    if 'learning_rate' in train_history.history:
        train_history.history['lr'] = train_history.history.pop('learning_rate')

    # Saving the Training History
    save_training_history(train_history, config)

    # Plotting the Training History
    plot_training_summary(config)

    # Training Summary
    training_time_elapsed = end - start
    print(f"[INFO] Total Time elapsed: {training_time_elapsed} seconds")
    print(f"[INFO] Time per epoch: {training_time_elapsed//config['epochs']} seconds")


if __name__ == "__main__":
    run()