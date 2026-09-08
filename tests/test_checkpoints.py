from pathlib import Path

from omnitcr.checkpoints import _resolve_checkpoint_path, _resolve_config_path
from omnitcr.config import BASE_CONFIG_FILE, TASK_CHECKPOINTS


def test_hugging_face_paths_match_published_folder_names():
    assert BASE_CONFIG_FILE == "OmniTCR(Base)/config.json"
    assert TASK_CHECKPOINTS == {
        "pm": "OmniTCR(FFT)_PM/model.safetensors",
        "pt": "OmniTCR(FFT)_PT/model.safetensors",
        "pmt": "OmniTCR(FFT)_PMT/model.safetensors",
        "pmab": "OmniTCR(FFT)_PMAB/model.safetensors",
        "repertoire": "OmniTCR(FFT)_CA/model.safetensors",
    }


def test_local_repository_root_is_resolved(tmp_path):
    config = tmp_path / BASE_CONFIG_FILE
    weights = tmp_path / TASK_CHECKPOINTS["pmt"]
    config.parent.mkdir(parents=True)
    weights.parent.mkdir(parents=True)
    config.write_text("{}")
    weights.write_bytes(b"test")

    resolved_config = _resolve_config_path(
        tmp_path, "unused", None, None, None
    )
    resolved_weights = _resolve_checkpoint_path(
        "pmt", tmp_path, "unused", None, None, None
    )
    assert resolved_config == Path(config)
    assert resolved_weights == Path(weights)

