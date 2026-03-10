import copy

import torch
import torch.nn as nn
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel

from TransformerForDiffusion import TransformerForDiffusion
from resnet import get_resnet, replace_bn_with_gn
from unet import ConditionalUnet1D


def _get_vlm_hidden_size(vlm_encoder):
    cfg = vlm_encoder.config
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    for key in ["text_config", "language_config", "llm_config"]:
        sub_cfg = getattr(cfg, key, None)
        if sub_cfg is not None and hasattr(sub_cfg, "hidden_size"):
            return int(sub_cfg.hidden_size)
    raise RuntimeError("Cannot infer hidden size from VLM config.")


def _from_pretrained_with_loading_info(model_cls, model_name, dtype):
    common = dict(trust_remote_code=True, output_loading_info=True)
    try:
        return model_cls.from_pretrained(model_name, dtype=dtype, **common)
    except TypeError:
        return model_cls.from_pretrained(model_name, torch_dtype=dtype, **common)


def _load_vlm_encoder(model_name, dtype):
    candidates = [
        getattr(transformers, "AutoModelForImageTextToText", None),
        getattr(transformers, "AutoModelForVision2Seq", None),
        AutoModel,
    ]
    last_err = None
    for cls in candidates:
        if cls is None:
            continue
        try:
            model, loading_info = _from_pretrained_with_loading_info(cls, model_name, dtype)
            missing = loading_info.get("missing_keys", []) if isinstance(loading_info, dict) else []
            if len(missing) > 128:
                raise RuntimeError(
                    f"VLM checkpoint appears partially initialized with {len(missing)} missing keys. "
                    "This usually means an incompatible model class/version was used."
                )
            return model
        except Exception as e:
            last_err = e
            continue

    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if "Qwen3-VL" in model_name and ver < (4, 50):
        raise RuntimeError(
            f"{model_name} may require newer transformers than {transformers.__version__}. "
            f"Please upgrade transformers or use an older VLM checkpoint."
        ) from last_err
    raise RuntimeError(f"Failed to load VLM encoder for {model_name}: {last_err}")


def build_flow_libero_vlm_model(args, device, action_dim, eval_state_dict=None):
    vlm_encoder = _load_vlm_encoder(args.vlm_model, dtype=torch.bfloat16)
    vlm_encoder = vlm_encoder.to(device, dtype=torch.bfloat16)

    if not args.frozen_vlm:
        vlm_lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=args.vlm_lora_r,
            lora_alpha=args.vlm_lora_alpha,
            lora_dropout=args.vlm_lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        vlm_encoder = get_peft_model(vlm_encoder, vlm_lora_config)

    vlm_embed_dim = _get_vlm_hidden_size(vlm_encoder)
    per_timestep_cond_dim = vlm_embed_dim + 8
    if args.net == "ConditionalUnet1D":
        global_cond_dim = per_timestep_cond_dim * args.obs_horizon
    elif args.net == "TransformerForDiffusion":
        global_cond_dim = per_timestep_cond_dim
    else:
        raise ValueError("net not found")

    if args.net == "TransformerForDiffusion":
        noise_pred_net = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=args.pred_horizon,
            cond_dim=global_cond_dim,
        )
    elif args.net == "ConditionalUnet1D":
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
        )

    nets = nn.ModuleDict(
        {
            "vlm_encoder": vlm_encoder,
            "noise_pred_net": noise_pred_net,
        }
    ).to(device, dtype=torch.bfloat16)

    if args.frozen_vlm:
        nets["vlm_encoder"].eval()
        for p in nets["vlm_encoder"].parameters():
            p.requires_grad = False
    elif train_with_vlm_lora and hasattr(nets["vlm_encoder"], "print_trainable_parameters"):
        nets["vlm_encoder"].print_trainable_parameters()

    return nets, use_vlm_lora


def build_flow_libero_unet_qwen_cot_model(args, device, action_dim, tokenizer):
    vision_encoder = get_resnet("resnet18")
    vision_encoder = replace_bn_with_gn(vision_encoder)

    text_encoder = AutoModel.from_pretrained(
        args.text_model,
        torch_dtype=torch.bfloat16,
    )
    text_encoder.resize_token_embeddings(len(tokenizer))
    target_text_encoder = copy.deepcopy(text_encoder)

    if not args.frozen_text_model:
        text_lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=args.text_lora_r,
            lora_alpha=args.text_lora_alpha,
            lora_dropout=args.text_lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        text_encoder = get_peft_model(text_encoder, text_lora_config)

    text_embed_dim = int(text_encoder.config.hidden_size)
    per_timestep_obs_dim = 512 * 2 + 8 + (0 if args.disable_text_input else text_embed_dim)
    per_timestep_cond_dim = per_timestep_obs_dim + 2 * text_embed_dim
    if args.net == "ConditionalUnet1D":
        global_cond_dim = per_timestep_cond_dim * args.obs_horizon
    elif args.net == "TransformerForDiffusion":
        global_cond_dim = per_timestep_cond_dim
    else:
        raise ValueError("net not found")

    if args.net == "TransformerForDiffusion":
        noise_pred_net = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=args.pred_horizon,
            cond_dim=global_cond_dim,
        )
    else:
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
        )

    cot_depth_head = nn.Sequential(
        nn.Linear(per_timestep_obs_dim, per_timestep_obs_dim),
        nn.SiLU(),
        nn.Linear(per_timestep_obs_dim, text_embed_dim),
    )
    cot_eef_head = nn.Sequential(
        nn.Linear(per_timestep_obs_dim, per_timestep_obs_dim),
        nn.SiLU(),
        nn.Linear(per_timestep_obs_dim, text_embed_dim),
    )

    nets = nn.ModuleDict(
        {
            "vision_encoder": vision_encoder,
            "text_encoder": text_encoder,
            "target_text_encoder": target_text_encoder,
            "noise_pred_net": noise_pred_net,
            "cot_depth_head": cot_depth_head,
            "cot_eef_head": cot_eef_head,
        }
    ).to(device, dtype=torch.bfloat16)

    nets["target_text_encoder"].eval()
    for p in nets["target_text_encoder"].parameters():
        p.requires_grad = False

    if args.frozen_text_model:
        nets["text_encoder"].eval()
        for p in nets["text_encoder"].parameters():
            p.requires_grad = False
    elif hasattr(nets["text_encoder"], "print_trainable_parameters"):
        nets["text_encoder"].print_trainable_parameters()

    return nets
