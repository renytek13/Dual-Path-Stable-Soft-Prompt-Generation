import os
import numpy as np
from torch.cuda.amp import GradScaler, autocast

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import dassl
from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights
from dassl.utils.tools import mkdir_if_missing
from dassl.optim import build_optimizer, build_lr_scheduler

from .basedg import *
from utils.clip_part import *


_tokenizer = _Tokenizer()


class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        pos_nctx = cfg.TRAINER.DPSPG.N_CTX
        neg_nctx = cfg.TRAINER.DPSPG.N_CTX
        pos_ctx_init = cfg.TRAINER.DPSPG.POS_CTX_INIT
        neg_ctx_init = cfg.TRAINER.DPSPG.NEG_CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        pos_ctx_vectors, pos_prompt_prefix, pos_nctx = self.generate_text_features(cfg, pos_ctx_init, clip_model, n_cls)
        neg_ctx_vectors, neg_prompt_prefix, neg_nctx = self.generate_text_features(cfg, neg_ctx_init, clip_model, n_cls)

        print(f'Initial positive context: "{pos_prompt_prefix}"')
        print(f"Number of positive context words (tokens): {pos_nctx}")
        print(f'Initial negative context: "{neg_prompt_prefix}"')
        print(f"Number of negative context words (tokens): {neg_nctx}")

        self.pos_ctx = nn.Parameter(pos_ctx_vectors)  # to be optimized
        self.neg_ctx = nn.Parameter(neg_ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        pos_prompts = [pos_prompt_prefix + " " + name + "." for name in classnames]
        neg_prompts = [neg_prompt_prefix + " " + name + "." for name in classnames]

        pos_tokenized_prompts = torch.cat([clip.tokenize(p) for p in pos_prompts])
        neg_tokenized_prompts = torch.cat([clip.tokenize(p) for p in neg_prompts])
        
        with torch.no_grad():
            pos_embedding = clip_model.token_embedding(pos_tokenized_prompts).type(dtype)
            neg_embedding = clip_model.token_embedding(neg_tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("pos_token_prefix", pos_embedding[:, :1, :])  # SOS
        self.register_buffer("pos_token_suffix", pos_embedding[:, 1 + pos_nctx :, :])  # CLS, EOS
        self.register_buffer("neg_token_prefix", neg_embedding[:, :1, :])  # SOS
        self.register_buffer("neg_token_suffix", neg_embedding[:, 1 + neg_nctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.pos_nctx = pos_nctx
        self.neg_nctx = neg_nctx
        self.pos_tokenized_prompts = pos_tokenized_prompts  # torch.Tensor
        self.neg_tokenized_prompts = neg_tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.DPSPG.CLASS_TOKEN_POSITION

    def forward(self):
        pos_ctx = self.pos_ctx
        neg_ctx = self.neg_ctx

        if pos_ctx.dim() == 2:
            pos_ctx = pos_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        if neg_ctx.dim() == 2:
            neg_ctx = neg_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        pos_prefix = self.pos_token_prefix
        pos_suffix = self.pos_token_suffix
        neg_prefix = self.neg_token_prefix
        neg_suffix = self.neg_token_suffix

        pos_prompts = self.construct_prompts(pos_ctx, pos_prefix, pos_suffix)
        neg_prompts = self.construct_prompts(neg_ctx, neg_prefix, neg_suffix)

        return pos_prompts, neg_prompts
    
    def generate_text_features(self, cfg, ctx_init, clip_model, n_cls):
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        if ctx_init: # use given words to initialize context vectors 
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:   # random initialization
            if cfg.TRAINER.DPSPG.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        return ctx_vectors, prompt_prefix, n_ctx
    
    
class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.pos_tokenized_prompts = self.prompt_learner.pos_tokenized_prompts
        self.neg_tokenized_prompts = self.prompt_learner.neg_tokenized_prompts
        self.n_cls = self.prompt_learner.n_cls
        
        self.text_encoder = BaseTextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.pos_prompts = torch.tensor(0)
        self.neg_prompts = torch.tensor(0)

    def forward(self, image):
        pos_prompts, neg_prompts = self.prompt_learner()
        self.pos_prompts = pos_prompts
        self.neg_prompts = neg_prompts

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        pos_text_features = self.text_encoder(pos_prompts, self.pos_tokenized_prompts)
        pos_text_features = pos_text_features / pos_text_features.norm(dim=-1, keepdim=True)

        neg_text_features = self.text_encoder(neg_prompts, self.neg_tokenized_prompts)
        neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        pos_logits = logit_scale * image_features @ pos_text_features.t()
        neg_logits = logit_scale * image_features @ neg_text_features.t()

        return pos_logits, neg_logits
    
    
@TRAINER_REGISTRY.register()
class DPSPG_CoOp(BaseDG):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)
        if not cfg.TEST.NO_TEST:
            self.test_best_result = -np.inf

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})...")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.DPSPG.PREC == "fp32" or cfg.TRAINER.DPSPG.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {sorted(enabled)}")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.DPSPG.PREC == "amp" else None
    
    def forward_backward(self, batch):
        images, labels = self.parse_batch_train(batch)

        cfg = self.cfg
        
        prec = self.cfg.TRAINER.DPSPG.PREC
        if prec == "amp":
            with autocast():
                output = self.model(images)
                loss = F.cross_entropy(output, labels)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            pos_logits, neg_logits = self.model(images)

            pos_loss = F.cross_entropy(pos_logits, labels)

            pos_labels = F.one_hot(labels, num_classes=self.n_cls).float()
            neg_labels = 1 - pos_labels

            pos_weight = torch.tensor([self.n_cls - 1.0], dtype=torch.float32).to(images.device)
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels, pos_weight=pos_weight)

            loss = pos_loss + neg_loss
            
            self.model_backward_and_update(loss)

        loss_summary = {
            "pos_loss": pos_loss.item(),
            "neg_loss": neg_loss.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        meet_checkpoint_freq = ((self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 
                                if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        curr_result = self.test(split="val")
        is_best = curr_result > self.best_result
        if is_best:
            self.best_result = curr_result
            self.best_epoch = self.epoch
            if self.cfg.SAVE_MODEL:
                self.save_model(self.epoch, self.output_dir, model_name="model-best.pth.tar")
                
                pos_prompt_dir = 'prompt_labels/' + self.cfg.DATASET.NAME.split('_')[0] + '/' + self.cfg.MODEL.BACKBONE.NAME.replace('/', '') + '/pos'
                mkdir_if_missing(pos_prompt_dir)
                neg_prompt_dir = 'prompt_labels/' + self.cfg.DATASET.NAME.split('_')[0] + '/' + self.cfg.MODEL.BACKBONE.NAME.replace('/', '') + '/neg'
                mkdir_if_missing(neg_prompt_dir)
                
                pos_prompts_path = os.path.join(pos_prompt_dir, f'{self.cfg.DATASET.NAME}_{self.cfg.TARGET_DOMAIN}.pt')
                pos_prompts = self.model.pos_prompts[0]
                pos_prompt = pos_prompts[1 : self.model.prompt_learner.pos_nctx+1, :]
                torch.save(pos_prompt, pos_prompts_path)

                neg_prompts_path = os.path.join(neg_prompt_dir, f'{self.cfg.DATASET.NAME}_{self.cfg.TARGET_DOMAIN}.pt')
                neg_prompts = self.model.neg_prompts[0]
                neg_prompt = neg_prompts[1 : self.model.prompt_learner.neg_nctx+1, :]
                torch.save(neg_prompt, neg_prompts_path)

                print(f'Positive prompt saved to {pos_prompts_path}')
                print(f'Negative prompt saved to {neg_prompts_path}')
        print('Domain {} val best acc: {:.1f}%, best epoch: {}'.format(self.cfg.TARGET_DOMAIN, self.best_result, self.best_epoch+1))

        self.set_model_mode("train")
        if self.cfg.SAVE_MODEL and (meet_checkpoint_freq or last_epoch):
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader
        print(f"Evaluate on the *{split}* set")
        
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)

            pos_logits, neg_logits = self.model(input)

            logits = pos_logits - neg_logits

            self.evaluator.process(logits, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return results['accuracy']
