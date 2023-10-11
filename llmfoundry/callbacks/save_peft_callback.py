import os
import tempfile
from typing import Any

from composer.core import Callback
from composer.utils import dist
from composer.utils.object_store.gcs_object_store import GCSObjectStore
from retrying import retry


class SavePeftCallback(Callback):
    def __init__(self, save_folder: str, **kwargs: Any):
        bucket, path = save_folder.removeprefix("gs://").split("/", 1)
        self.store = GCSObjectStore(bucket, path)

        super().__init__(**kwargs)

    def epoch_checkpoint(self, state, logger):
        if dist.get_global_rank() != 0:
            return

        try:
            path_prefix = f"adapter_ep{state.timestamp.epoch.value}"

            with tempfile.TemporaryDirectory() as tmpdir:
                peft_model = state.model.model
                peft_model.save_pretrained(tmpdir)

                for filename in os.listdir(tmpdir):
                    local_path = os.path.join(tmpdir, filename)
                    remote_path = os.path.join(path_prefix, filename)
                    self._upload_object(remote_path, local_path)
        except Exception as e:
            logger.warning(f"Failed to save PEFT model: {e}")

    @retry(
        stop_max_attempt_number=5,
        wait_random_min=5000,
        wait_random_max=10000,
    )
    def _upload_object(self, remote_path: str, local_path: str):
        self.store.upload_object(remote_path, local_path)
