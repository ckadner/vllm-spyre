from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from vllm.logger import init_logger

from vllm_spyre import envs as envs_spyre

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

_config_file = Path(__file__).parent / "supported_configurations.yaml"

logger = init_logger(__name__)

# warmup_shape = [prompt_length, new_tokens, batch_size]
WarmupShapes = list[tuple[int, int, int]] | list[list[int]]

@dataclass(order=True)
class RuntimeConfiguration:
    cb: bool = False
    tp_size: int = 1
    max_seq_len: int = 0
    max_num_seqs: int = 0
    max_prompt: int | None = field(compare=False, default=None)


@dataclass
class ModelRuntimeConfiguration:
    model: str
    configs: list[RuntimeConfiguration] | None = None
    ignore: bool = False

    def __post_init__(self):
        self.configs = [
            RuntimeConfiguration(**cfg) if isinstance(cfg, dict) else cfg
            for cfg in self.configs or []
        ]


model_runtime_configs: list[ModelRuntimeConfiguration] | None = None
ignored_models: set[str] = set()
runtime_configs_by_model: dict[str, list[RuntimeConfiguration]]


def load_config_yaml() -> list[dict[str, Any]]:
    with open(_config_file, encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data


def initialize_supported_configurations(yaml_data: list[dict[str, Any]]):
    global model_runtime_configs, ignored_models, runtime_configs_by_model
    model_runtime_configs = [
        ModelRuntimeConfiguration(**config_dict) for config_dict in yaml_data
    ]
    ignored_models = {mrc.model for mrc in model_runtime_configs if mrc.ignore}
    runtime_configs_by_model = {
        mrc.model: mrc.configs or []
        for mrc in model_runtime_configs if not mrc.ignore
    }


def initialize_supported_configurations_from_file():
    yaml_data = load_config_yaml()
    initialize_supported_configurations(yaml_data)


def verify_config_parameters(c: RuntimeConfiguration) -> bool:
    found_invalid_parameters = False

    def verify(msg: str, is_valid: bool):
        nonlocal found_invalid_parameters
        if not is_valid:
            found_invalid_parameters = True
            logger.warning(msg)

    def is_power_of_2(n: int) -> bool:
        return (n > 0) and (n & (n - 1) == 0)

    verify(f"'tensor_parallel_size' must be a power of 2, found {c.tp_size}",
           is_power_of_2(c.tp_size))

    verify(
        f"'max_seq_len' must be a multiple of 64,"
        f" found {c.max_seq_len}", c.max_seq_len % 64 == 0)
    verify(
        f"'max_num_seqs' must be a power of 2,"
        f" found {c.max_num_seqs}", is_power_of_2(c.max_num_seqs))

    if not c.cb:
        verify(
            f"'max_prompt' must be a multiple of 64,"
            f" found {c.max_prompt}", c.max_prompt % 64 == 0)
        verify(
            f"'max_prompt' must be a <= 'max_seq_len',"
            f" found {c.max_prompt}", c.max_prompt <= c.max_seq_len)

        if c.max_prompt == 0:
            c.max_prompt = c.max_seq_len

    return not found_invalid_parameters


def validate_runtime_configuration(
        vllm_config: VllmConfig,
        warmup_shapes: WarmupShapes | None = None) -> bool:
    """
    Verify if the requested model and configuration is supported by comparing
    the requested configuration to all the supported configurations.
    """
    model         = vllm_config.model_config.model
    tp_size       = vllm_config.parallel_config.tensor_parallel_size,
    max_model_len = vllm_config.model_config.max_model_len,
    max_num_seqs  = vllm_config.scheduler_config.max_num_seqs,
    # we only validate runtime configurations when running on Spyre cards
    if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND != "sendnn":
        logger.info(
            "Model and runtime configuration validation bypassed for"
            " backend '%s'", envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND)
        return True

    global model_runtime_configs
    if model_runtime_configs is None:
        initialize_supported_configurations_from_file()

    if model in ignored_models:
        logger.info("Model '%s' is ignored", model)
        return True

    if model not in runtime_configs_by_model:
        logger.warning("Model '%s' is not supported", model)
        return False

    use_cb = envs_spyre.VLLM_SPYRE_USE_CB

    supported_configs = runtime_configs_by_model.get(model, [])

    matching_configs = [c for c in supported_configs if c.cb == use_cb and c.tp_size == tp_size]

    max_prompt_len = 0
    if not use_cb:
        warmup_shapes = warmup_shapes or []
        max_prompt_len = max(s[0] for s in warmup_shapes)

    requested_config = RuntimeConfiguration(
        cb=use_cb,
        tp_size=tp_size,
        max_seq_len=max_model_len, # set in platform for CB and SB
        max_num_seqs=max_num_seqs, # set in platform for CB and SB
        max_prompt=max_prompt_len)

    if not verify_config_parameters(requested_config):
        return False

    if len(matching_configs) == 0:
        logger.warning(
            "The requested configuration is not supported for"
            " model '%s': %s", model, str(requested_config))
        return False
    else:
        logger.info(
            "The requested configuration is supported for"
            " model '%s': %s", model, str(requested_config))
        if len(matching_configs) > 1:
            logger.warn(
                "More than one matching configuration was for"
                " model '%s': %s", model, str(requested_config))
        
    supported_config = matching_configs[0]

    msg = ""

    if not use_cb and requested_config.max_prompt > supported_config.max_prompt:
        msg = "The requested prompt length"
    elif requested_config.max_seq_len > supported_config.max_seq_len:
        msg = "The requested total sequence length"
    elif requested_config.max_num_seqs > supported_config.max_num_seqs:
        msg = "The requested number of sequences"

    if msg:
        logger.warning(
            " is not supported %s for"
            " model '%s': %s", msg, model, str(requested_config))
        return False

    return True