import os
import tempfile
from typing import Any

from composer.core import Callback
from composer.utils.object_store.gcs_object_store import GCSObjectStore


class SavePeftCallback(Callback):
    def __init__(self, save_folder: str, **kwargs: Any):
        bucket, path = save_folder.removeprefix("gs://").split("/", 1)
        self.store = GCSObjectStore(bucket, path)

        super().__init__(**kwargs)

    def epoch_checkpoint(self, state, logger):
        try:
            path_prefix = f"adapter_ep{state.timestamp.epoch.value}"

            with tempfile.TemporaryDirectory() as tmpdir:
                peft_model = state.model.model
                peft_model.save_pretrained(tmpdir)

                for filename in os.listdir(tmpdir):
                    local_path = os.path.join(tmpdir, filename)
                    remote_path = os.path.join(path_prefix, filename)
                    self.store.upload_object(remote_path, local_path)
        except Exception:
            logger.exception("Failed to save PEFT model")
