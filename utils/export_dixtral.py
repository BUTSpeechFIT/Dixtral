"""Merge a LoRA-finetuned Dixtral checkpoint into the base model and push it to the HF Hub.

Mirrors the flow of ``export_dicow.py``: rebuild the model, load the trained weights,
register it for ``trust_remote_code`` auto-loading, then push the model, processor and a
README from ``export_sources/readmes/{model_name}.md``.

A ``train_dixtral.py`` LoRA checkpoint stores weights in two pieces:
  * ``non_peft_params.safetensors`` -- trained non-LoRA params (FDDT, audio layers, projector);
    keys already have the ``base_model.model.`` prefix stripped, so they match the raw model.
  * ``adapter_model.safetensors`` (+ ``adapter_config.json``) -- the LoRA adapter on the LLM.

Example
-------
    source configs/local_paths_2.sh

    # QA model (question answering)
    python utils/export_dixtral.py --model_path /data/user_data/apolok/QA_FT  --model_name Dixtral_QA
    # TS-ASR transcription model
    python utils/export_dixtral.py --model_path /data/user_data/apolok/LORA_18k --model_name Dixtral_TS-ASR
"""

import os
from types import SimpleNamespace

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from safetensors.torch import load_file

from models.container import DixtralContainer
from models.dixtral.configuration_dixtral import DixtralConfig
from models.dixtral.modeling_dixtral import DixtralForConditionalGeneration


def argparse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the LoRA training checkpoint (with non_peft_params + adapter_*)')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model on the Hub')
    parser.add_argument('--org', type=str, default='BUT-FIT', help='Hugging Face organization name')
    parser.add_argument('--base_model', type=str, default='mistralai/Voxtral-Mini-3B-2507')
    parser.add_argument('--private', action='store_true', help='Create a private repo')
    return parser.parse_args()


if __name__ == '__main__':
    args = argparse()
    NEW_MODEL_ID = f"{args.org}/{args.model_name}"
    api = HfApi()

    # Model args matching how QA_FT / LORA_18k were trained: Voxtral-Mini-3B with the DiCoW
    # encoder replaced from $SRC_ROOT/dicow_large_v3, no soft prompts, no CTC head.
    model_args = SimpleNamespace(
        dixtral_base_model=args.base_model,
        dixtral_replace_encoder_from=os.path.join(os.environ['SRC_ROOT'], 'dicow_large_v3'),
        dixtral_load_fddt_from=None,
        num_soft_prompts=0,
        ctc_weight=0,
    )

    # 1. Rebuild the base model the same way the container does (no LoRA wrapper here).
    container = DixtralContainer(model_args=model_args, use_lora=False)
    model = container.model

    # 2. Load the trained non-PEFT params (FDDT / audio layers / projector).
    non_peft_sd = load_file(os.path.join(args.model_path, 'non_peft_params.safetensors'))
    missing, unexpected = model.load_state_dict(non_peft_sd, strict=False)
    assert not unexpected, f"Unexpected non_peft keys (config mismatch): {unexpected[:5]}"

    # 3. Attach the saved LoRA adapter and merge it into the base weights.
    model = PeftModel.from_pretrained(model, args.model_path).merge_and_unload()
    model = model.to(torch.bfloat16)

    # 4. Register custom classes so the repo loads with trust_remote_code=True.
    DixtralConfig.register_for_auto_class()
    DixtralForConditionalGeneration.register_for_auto_class("AutoModel")

    # 5. Push model + processor + README.
    model.push_to_hub(NEW_MODEL_ID, private=args.private)
    container.processor.push_to_hub(NEW_MODEL_ID, private=args.private)
    api.upload_file(
        path_or_fileobj=f"{os.environ['SRC_ROOT']}/export_sources/readmes/{args.model_name}.md",
        path_in_repo="README.md",
        repo_id=NEW_MODEL_ID,
    )
    print(f"Done. https://huggingface.co/{NEW_MODEL_ID}")
