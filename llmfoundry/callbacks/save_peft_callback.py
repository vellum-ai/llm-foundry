import os
import shutil
from typing import Any

from composer.core import Callback
from composer.utils.object_store.gcs_object_store import GCSObjectStore


class SavePeftCallback(Callback):
    def __init__(self, save_folder: str, **kwargs: Any):
        bucket, path = save_folder.removeprefix("gs://").split("/", 1)
        self.store = GCSObjectStore(bucket, path)

        super().__init__(**kwargs)

    def epoch_checkpoint(self, state, logger):
        save_folder = f"adapter_ep{state.timestamp.epoch.value}"

        peft_model = state.model.model
        peft_model.save_pretrained(save_folder)

        for filename in os.listdir(save_folder):
            path = os.path.join(save_folder, filename)
            self.store.upload_object(path, path)

        # Remove the folder
        shutil.rmtree(save_folder)
