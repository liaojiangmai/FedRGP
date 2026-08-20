import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'FedRGP',
                      "vision_depth": cfg.TRAINER.FEDRGP.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.FEDRGP.PROMPT_DEPTH_TEXT, 
                      "vision_ctx": cfg.TRAINER.FEDRGP.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.FEDRGP.N_CTX_TEXT,
                      "vision_num_prompts": cfg.TRAINER.FEDRGP.NUM_PROMPTS_VISION,
                      "language_num_prompts": cfg.TRAINER.FEDRGP.NUM_PROMPTS_TEXT}
    
    # Ensure vision prompt depth is at least 1 when using vision prompts
    if design_details["vision_num_prompts"] > 0 and design_details["vision_depth"] <= 0:
        design_details["vision_depth"] = 1
        
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiPromptLearner(nn.Module):
    """Dual-stream prompt learner for clean/noise experts."""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        assert cfg.TRAINER.FEDRGP.PROMPT_DEPTH_TEXT >= 1, "In FedRGP, Language prompt depth should be >=1"

        n_ctx = cfg.TRAINER.FEDRGP.N_CTX_TEXT
        ctx_init = cfg.TRAINER.FEDRGP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.num_streams = 2
        self.num_prompts_text = max(self.num_streams, int(cfg.TRAINER.FEDRGP.NUM_PROMPTS_TEXT))

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

            clean_ctx = ctx_vectors.clone()
            torch.manual_seed(123)
            noise_ctx = clean_ctx + torch.randn_like(clean_ctx) * 0.01
            ctx_vectors = torch.stack([clean_ctx, noise_ctx], dim=0)
        else:
            ctx_vectors = torch.empty(self.num_streams, n_ctx, ctx_dim, dtype=dtype)
            for i in range(self.num_streams):
                torch.manual_seed(i + 42)
                nn.init.normal_(ctx_vectors[i], std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print("FedRGP dual-stream prompts")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print("Prompt streams: clean + noise")
        print(f"Number of vision prompts: {cfg.TRAINER.FEDRGP.NUM_PROMPTS_VISION}")

        # Shape: [2, n_ctx, ctx_dim]
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )

        return prompts

    def build_prompts_from_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0)
        if ctx.size(0) == 1:
            ctx = ctx.expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        return self.construct_prompts(ctx, prefix, suffix)

    def forward(self):
        clean_ctx = self.ctx[0]
        noise_ctx = self.ctx[1] if self.ctx.size(0) > 1 else self.ctx[0]
        clean_prompts = self.build_prompts_from_ctx(clean_ctx)
        noise_prompts = self.build_prompts_from_ctx(noise_ctx)
        return clean_prompts, noise_prompts


class CustomCLIP(nn.Module):
    """CLIP model with dual-stream clean/noise prompts for FedRGP."""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = [name.replace("_", " ") for name in classnames]

        # Dual-stream indices
        self.clean_index = 0
        self.noise_index = 1

        # RGP config
        self.rgp_enabled = getattr(cfg.TRAINER.FEDRGP, 'RGP_ENABLED', True)
        self.rgp_tau = float(getattr(cfg.TRAINER.FEDRGP, 'RGP_TAU', 1.0))
        self.rgp_consistency_weight = float(getattr(cfg.TRAINER.FEDRGP, 'RGP_CONSIST_WEIGHT', 0.5))

        # Global view prompts for credibility and consistency
        self.global_view_text = None
        self.global_view_vision = None

    def set_active_prompts(self, text_count: int, vision_count: int) -> None:
        # No-op for dual-stream setup; keep for API compatibility
        return

    def set_global_view(self, text_prompts: torch.Tensor, vision_prompts: torch.Tensor) -> None:
        self.global_view_text = text_prompts
        self.global_view_vision = vision_prompts

    def _build_text_features_from_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        if ctx.dim() == 3:
            ctx = ctx[0]
        prompts = self.prompt_learner.build_prompts_from_ctx(ctx)
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _compute_credibility(self, image, label):
        if self.global_view_text is None or label is None:
            return None
        with torch.no_grad():
            # Image features (use clean vision prompt if available)
            if hasattr(self.image_encoder, 'VPT'):
                image_features = self.image_encoder(image.type(self.dtype), prompt_idx=self.clean_index)
            else:
                image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            global_text_features = self._build_text_features_from_ctx(self.global_view_text.to(image.device))
            label_features = global_text_features[label]
            sims = (image_features * label_features).sum(dim=-1)
            if self.rgp_tau > 0:
                sims = sims / self.rgp_tau
            sims = torch.relu(sims)

            if sims.numel() == 0:
                return None

            minv = sims.min()
            maxv = sims.max()
            if (maxv - minv) > 1e-6:
                weights = (sims - minv) / (maxv - minv + 1e-8)
            else:
                weights = torch.full_like(sims, 0.5)

        return weights.detach()

    def forward(self, image, label=None, image_paths=None):
        clean_prompts, noise_prompts = self.prompt_learner()

        text_features_clean = self.text_encoder(clean_prompts, self.tokenized_prompts)
        text_features_clean = text_features_clean / text_features_clean.norm(dim=-1, keepdim=True)
        text_features_noise = self.text_encoder(noise_prompts, self.tokenized_prompts)
        text_features_noise = text_features_noise / text_features_noise.norm(dim=-1, keepdim=True)

        if hasattr(self.image_encoder, 'VPT'):
            image_features_clean = self.image_encoder(image.type(self.dtype), prompt_idx=self.clean_index)
            image_features_noise = self.image_encoder(image.type(self.dtype), prompt_idx=self.noise_index)
        else:
            image_features_clean = self.image_encoder(image.type(self.dtype))
            image_features_noise = image_features_clean

        image_features_clean = image_features_clean / image_features_clean.norm(dim=-1, keepdim=True)
        image_features_noise = image_features_noise / image_features_noise.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_clean = logit_scale * image_features_clean @ text_features_clean.t()
        logits_noise = logit_scale * image_features_noise @ text_features_noise.t()

        if self.training and label is not None and self.rgp_enabled:
            weights = self._compute_credibility(image, label)
        else:
            weights = None

        if weights is not None:
            w = weights.view(-1, 1)
            final_logits = w * logits_clean + (1.0 - w) * logits_noise
        else:
            # Inference or fallback: use clean expert only
            final_logits = logits_clean

        if self.training and label is not None:
            ce_loss = F.cross_entropy(final_logits, label)
            consistency_loss = torch.tensor(0.0, device=image.device, dtype=self.dtype)

            if self.rgp_consistency_weight > 0 and self.global_view_text is not None:
                with torch.no_grad():
                    global_text_features = self._build_text_features_from_ctx(self.global_view_text.to(image.device))
                consistency_loss = F.mse_loss(text_features_clean, global_text_features)

            total_loss = ce_loss + self.rgp_consistency_weight * consistency_loss
            return total_loss, final_logits, consistency_loss

        return final_logits


# @TRAINER_REGISTRY.register()
class FedRGP(TrainerX):
    """Federated Multi-Grained Prompting trainer.

    Supports multiple vision and text prompts with selective aggregation.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.FEDRGP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.FEDRGP.PREC == "fp32" or cfg.TRAINER.FEDRGP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP with multiple prompts")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")

        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        for name, param in self.model.named_parameters():
            if "prompt_learner.ctx" in name:
                param.requires_grad_(True)
                print(f"Enabling gradient: {name}, shape={param.shape}")

            if "visual.VPT" in name or "image_encoder.VPT" in name:
                param.requires_grad_(True)
                print(f"Enabling gradient: {name}, shape={param.shape}")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.FEDRGP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        print("\nTrainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}, shape={param.shape}")

    def set_active_prompts(self, text_count: int, vision_count: int) -> None:
        """Set active prompt group counts for adaptive prompt growth."""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(model, "set_active_prompts"):
            model.set_active_prompts(text_count, vision_count)

    def set_global_view(self, text_prompts: torch.Tensor, vision_prompts: torch.Tensor) -> None:
        """Set global view prompts for guidance-based training."""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(model, "set_global_view"):
            model.set_global_view(text_prompts, vision_prompts)
                
    def forward_backward(self, batch):
        """Forward and backward pass for training."""
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.FEDRGP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image, label)
                loss = output[0]  # Total loss is the first output
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, label)
            loss = output[0]
            self.model_backward_and_update(loss)

        def get_loss_value(loss_tensor_or_scalar):
            if hasattr(loss_tensor_or_scalar, 'item'):
                return loss_tensor_or_scalar.item()
            return float(loss_tensor_or_scalar)

        loss_summary = {
            "loss": get_loss_value(loss),
            "acc": compute_accuracy(output[1], label)[0].item(),
        }

        if len(output) >= 3:
            consistency_loss = output[2]
            if consistency_loss is not None:
                loss_summary["consistency_loss"] = get_loss_value(consistency_loss)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

            has_vpt_params = False
            for name, param in self.model.named_parameters():
                if ("visual.VPT" in name or "image_encoder.VPT" in name) and param.requires_grad:
                    has_vpt_params = True
                    param_norm = torch.norm(param.data)

                    if hasattr(self, 'old_vpt_params') and name in self.old_vpt_params:
                        param_diff = torch.norm(param.data - self.old_vpt_params[name])
                        change_percent = param_diff / torch.norm(self.old_vpt_params[name]) * 100 if torch.norm(self.old_vpt_params[name]) > 0 else 0
                        print(f"VPT param '{name}' change: {param_diff:.6f} ({change_percent:.2f}%)")

                    if not hasattr(self, 'old_vpt_params'):
                        self.old_vpt_params = {}
                    self.old_vpt_params[name] = param.data.clone().detach()

            if not has_vpt_params:
                print("Warning: No trainable VPT parameters found!")

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def model_inference(self, input):
        """Model inference, returns logits only."""
        output = self.model(input)
        if isinstance(output, tuple):
            return output[0]
        return output

    def load_model(self, directory, epoch=None):
        """Load pretrained model, ignoring fixed token vectors."""
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # Load best model by default
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

    def test(self, split=None, is_global=False, current_epoch=0, idx=-1, global_test=False):
        """Test pipeline with visualization support."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and hasattr(self, 'val_loader') and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"

            if global_test and not getattr(self, 'is_special_dataset', False):
                print(f"Global test mode: using global test set")
                if hasattr(self, 'test_loader') and self.test_loader is not None:
                    data_loader = self.test_loader
                elif hasattr(self, 'fed_test_loader_x_dict') and len(self.fed_test_loader_x_dict) > 0:
                    for client_id in sorted(self.fed_test_loader_x_dict.keys()):
                        if self.fed_test_loader_x_dict[client_id] is not None:
                            print(f"Warning: test_loader not found, using client {client_id}'s federated test set")
                            data_loader = self.fed_test_loader_x_dict[client_id]
                            break
                else:
                    raise ValueError("No available test dataset")
            elif idx != -1:
                print(f"Client test mode: using federated test set for client {idx}")
                data_loader = None
                if hasattr(self, 'fed_test_loader_dict') and idx in self.fed_test_loader_dict:
                    data_loader = self.fed_test_loader_dict[idx]
                elif hasattr(self, 'fed_test_loader_x_dict') and idx in self.fed_test_loader_x_dict:
                    data_loader = self.fed_test_loader_x_dict[idx]

                if data_loader is None:
                    print(f"Warning: Client {idx} has no dedicated test set, trying global test set")
                    if hasattr(self, 'test_loader') and self.test_loader is not None:
                        data_loader = self.test_loader
                    else:
                        raise ValueError(f"No test data available for client {idx}")
            else:
                print(f"Standard test mode: using global test set")
                if hasattr(self, 'test_loader') and self.test_loader is not None:
                    data_loader = self.test_loader
                elif hasattr(self, 'fed_test_loader_x_dict') and len(self.fed_test_loader_x_dict) > 0:
                    for client_id in sorted(self.fed_test_loader_x_dict.keys()):
                        if self.fed_test_loader_x_dict[client_id] is not None:
                            print(f"Warning: test_loader not found, using client {client_id}'s federated test set")
                            data_loader = self.fed_test_loader_x_dict[client_id]
                            break
                else:
                    raise ValueError("No available test dataset")

        test_type = "global" if global_test and not getattr(self, 'is_special_dataset', False) else f"client {idx}"
        print(f"Evaluating on {test_type} {split} set")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)

            with torch.no_grad():
                output = self.model_inference(input)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            if not is_global and idx >= 0:
                tag = f"{tag}/{str(idx)}"
            elif global_test and not getattr(self, 'is_special_dataset', False):
                tag = f"{tag}/global"
            self.write_scalar(tag, v, current_epoch)

        return list(results.values()) 
