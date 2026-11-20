"""CPU-only: structural + config checks of a local fixed checkpoint against its base (inside the quant eval image)."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/tools")
from local_eval import headers
from core.models.quant_models import QuantFixedSpec
from validator.evaluation.quant_structure import TensorSet, TensorSpec, check_fixed_submission, expected_fixed_tensor_set
from validator.evaluation.quant_config_check import quant_config_failure, config_drift_failure
from validator.evaluation.quant_vllm import resolved_model_config

base, sub, spec_json = sys.argv[1], sys.argv[2], sys.argv[3]
spec = QuantFixedSpec.model_validate_json(spec_json)
to_set = lambda h: TensorSet(tensors={k: TensorSpec(**v) for k, v in h.items()})
bc = json.loads((Path(base) / "config.json").read_text()); sc = json.loads((Path(sub) / "config.json").read_text())
print("structural:", check_fixed_submission(to_set(headers(Path(sub))), expected_fixed_tensor_set(to_set(headers(Path(base))), bc, spec)) or "ok")
print("quant_config:", quant_config_failure(sc, spec) or "ok")
print("drift:", config_drift_failure(resolved_model_config(sc), resolved_model_config(bc)) or "ok")
