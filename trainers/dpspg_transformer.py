import datetime
import pickle
import random
import time
import os
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from tqdm import *
from utils.clip_part import load_clip_to_cpu, BaseTextEncoder
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import dassl
from dassl.utils.meters import AverageMeter, MetricMeter
from dassl.utils.tools import mkdir_if_missing
from dassl.utils.torchtools import partial
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

_tokenizer = _Tokenizer()

class Encoder(nn.Module):
    def __init__(self, cfg, n_ctx):
        super(Encoder, self).__init__()

        self.cfg = cfg
        self.n_ctx = n_ctx

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=cfg.FEAT_DIM, nhead=8),
            num_layers=4
        )
        self.fc = nn.Linear(in_features=cfg.FEAT_DIM, out_features=n_ctx * 512)


    def forward(self, image):
        gen_input = image.view(image.shape[0], -1)
        gen_input = gen_input.unsqueeze(0)  # add a dimension for sequence length
        gen_ctx_prompt = self.transformer_encoder(gen_input)

        gen_ctx_prompt = gen_ctx_prompt.squeeze(0)
        gen_ctx_prompt = self.fc(gen_ctx_prompt)
        gen_ctx_prompt = gen_ctx_prompt.view(-1, self.n_ctx, 512)

        return gen_ctx_prompt


@TRAINER_REGISTRY.register()
class DPSPG_TRANSFORMER(TrainerX):
    """Positive-Negative Soft Prompt Generation with Transformer-based Model (DPSPG).
    """    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # loading CLIP model
        self.clip_model = load_clip_to_cpu(cfg).to(self.device)

        self.pos_nctx = cfg.TRAINER.SPG.N_CTX
        self.neg_nctx = cfg.TRAINER.SPG.N_CTX
        self.pos_ctx_init = cfg.TRAINER.SPG.POS_CTX_INIT
        self.neg_ctx_init = cfg.TRAINER.SPG.NEG_CTX_INIT
        if self.pos_ctx_init:
            self.pos_ctx_init = self.pos_ctx_init.replace("_", " ")
            self.pos_nctx = len(self.pos_ctx_init.split(" "))
        if self.neg_ctx_init:
            self.neg_ctx_init = self.neg_ctx_init.replace("_", " ")
            self.neg_nctx = len(self.neg_ctx_init.split(" "))
        
        print("Building SPG_TRANSFORMER")
        self.pmodel = Encoder(cfg, self.pos_nctx)
        self.nmodel = Encoder(cfg, self.neg_nctx)
        
        self.pmodel.to(self.device)
        self.nmodel.to(self.device)
        
        self.optimizer_P = torch.optim.AdamW(self.pmodel.parameters(), lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        self.optimizer_N = torch.optim.AdamW(self.nmodel.parameters(), lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        self.sched_P = build_lr_scheduler(self.optimizer_P, cfg.OPTIM)
        self.sched_N = build_lr_scheduler(self.optimizer_N, cfg.OPTIM)
        self.register_model("posencoder", self.pmodel, self.optimizer_P, self.sched_P)
        self.register_model("negencoder", self.nmodel, self.optimizer_N, self.sched_N)

        self.best_pos_prompts = {}
        self.best_neg_prompts = {}

        for i in range(len(cfg.ALL_DOMAINS)):
            pos_prompt_dir = 'prompt_labels' + '/' + self.cfg.DATASET.NAME.split('_')[1] + '/' + self.cfg.MODEL.BACKBONE.NAME.replace('/', '') + '/pos'
            neg_prompt_dir = 'prompt_labels' + '/' + self.cfg.DATASET.NAME.split('_')[1] + '/' + self.cfg.MODEL.BACKBONE.NAME.replace('/', '') + '/neg'

            pos_prompts_path = os.path.join(pos_prompt_dir, self.cfg.DATASET.NAME.split('_')[1] + '_CoOp_' + self.cfg.ALL_DOMAINS[i] + '.pt')
            neg_prompts_path = os.path.join(neg_prompt_dir, self.cfg.DATASET.NAME.split('_')[1] + '_CoOp_' + self.cfg.ALL_DOMAINS[i] + '.pt')
            
            pos_prompt_label = torch.load(pos_prompts_path).to(self.device)
            self.best_pos_prompts[i] = pos_prompt_label

            neg_prompt_label = torch.load(neg_prompts_path).to(self.device)
            self.best_neg_prompts[i] = neg_prompt_label

    def forward_backward(self, batch):
        cfg = self.cfg
        image, label, domain = self.parse_batch_train(batch)

        # select MSELoss as the loss function
        adversarial_loss = torch.nn.MSELoss()
        adversarial_loss.to(self.device)

        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.to(torch.float32)
        image_features = image_features.detach()


        self.pmodel.train()
        self.nmodel.train()

        
        """Train Pos & Neg Encoder
        """
        self.optimizer_P.zero_grad()
        self.optimizer_N.zero_grad()
        pos_gen_prompt = self.pmodel(image_features)
        neg_gen_prompt = self.nmodel(image_features)

        pos_single_batch_prompts = torch.unsqueeze(self.best_pos_prompts[0], 0)
        pos_batch_prompts = torch.clone(pos_single_batch_prompts)
        for imgnum in range(image.shape[0]):
            if(imgnum < image.shape[0]-1):
                pos_batch_prompts = torch.cat((pos_batch_prompts, pos_single_batch_prompts), dim=0)
            pos_batch_prompts[imgnum] = self.best_pos_prompts[int(domain[imgnum])]

        pos_batch_prompts = pos_batch_prompts.to(torch.float32)
        loss_pos = adversarial_loss(pos_gen_prompt, pos_batch_prompts)

        neg_single_batch_prompts = torch.unsqueeze(self.best_neg_prompts[0], 0)
        neg_batch_prompts = torch.clone(neg_single_batch_prompts)
        for imgnum in range(image.shape[0]):
            if(imgnum < image.shape[0]-1):
                neg_batch_prompts = torch.cat((neg_batch_prompts, neg_single_batch_prompts), dim=0)
            neg_batch_prompts[imgnum] = self.best_neg_prompts[int(domain[imgnum])]

        neg_batch_prompts = neg_batch_prompts.to(torch.float32)
        loss_neg = adversarial_loss(neg_gen_prompt, neg_batch_prompts)

        loss = loss_pos + loss_neg
        loss.backward()

        self.optimizer_P.step()
        self.optimizer_N.step()


        loss_summary = {
            "loss_pos": loss_pos.item(),
            "loss_neg": loss_neg.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def generate_text_features(self, ctx_init):
        n_ctx = self.cfg.TRAINER.SPG.N_CTX
        if ctx_init: # use given words to initialize context vectors 
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt_prefix = ctx_init
        else:   # random initialization
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        return prompt_prefix, n_ctx

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.pmodel.eval()
        self.nmodel.eval()
        self.evaluator.reset()

        cfg = self.cfg

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"\nEvaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)
            
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.to(torch.float32)
            text_encoder = BaseTextEncoder(self.clip_model)

            pos_ctx_init = cfg.TRAINER.SPG.POS_CTX_INIT
            neg_ctx_init = cfg.TRAINER.SPG.NEG_CTX_INIT
            dtype = self.clip_model.dtype

            pos_prompt_prefix, pos_nctx = self.generate_text_features(pos_ctx_init)
            neg_prompt_prefix, neg_nctx = self.generate_text_features(neg_ctx_init)

            classnames = self.dm.dataset.classnames
            n_cls = len(classnames)

            classnames = [name.replace("_", " ") for name in classnames]
            pos_prompts = [pos_prompt_prefix + " " + name + "." for name in classnames]
            neg_prompts = [neg_prompt_prefix + " " + name + "." for name in classnames]

            pos_tokenized_prompts = torch.cat([clip.tokenize(p) for p in pos_prompts])
            neg_tokenized_prompts = torch.cat([clip.tokenize(p) for p in neg_prompts])

            token_embedding = self.clip_model.token_embedding
            
            pos_tokenized_prompts = pos_tokenized_prompts.to(self.device)
            neg_tokenized_prompts = neg_tokenized_prompts.to(self.device)

            with torch.no_grad():
                pos_embedding = token_embedding(pos_tokenized_prompts).type(self.clip_model.dtype)
                neg_embedding = token_embedding(neg_tokenized_prompts).type(self.clip_model.dtype)
            self.pos_token_prefix = pos_embedding[:, :1, :]  # SOS
            self.pos_token_suffix = pos_embedding[:, 1 + pos_nctx :, :]  # CLS, EOS
            self.neg_token_prefix = neg_embedding[:, :1, :]  # SOS
            self.neg_token_suffix = neg_embedding[:, 1 + neg_nctx :, :]  # CLS, EOS

            self.class_token_position = self.cfg.TRAINER.SPG.CLASS_TOKEN_POSITION

            pos_prefix = self.pos_token_prefix.to(self.device)
            pos_suffix = self.pos_token_suffix.to(self.device)
            neg_prefix = self.neg_token_prefix.to(self.device)
            neg_suffix = self.neg_token_suffix.to(self.device)

            pos_goutput = self.pmodel(image_features)
            neg_goutput = self.nmodel(image_features)


            logits = []
            for pos_i, neg_i, img_i in zip(pos_goutput, neg_goutput, image_features):
                pos_i = pos_i.unsqueeze(0)
                neg_i = neg_i.unsqueeze(0)
                img_i = img_i.unsqueeze(0)

                pos_batch_prompts = torch.ones(n_cls, pos_i.size(1), pos_i.size(2)).to(self.device)
                for idnum in range(n_cls):
                    pos_batch_prompts[idnum] = torch.clone(pos_i[0])
                pos_gen_prompt = torch.tensor(pos_batch_prompts, dtype=torch.float16)

                neg_batch_prompts = torch.ones(n_cls, neg_i.size(1), neg_i.size(2)).to(self.device)
                for idnum in range(n_cls):
                    neg_batch_prompts[idnum] = torch.clone(neg_i[0])
                neg_gen_prompt = torch.tensor(neg_batch_prompts, dtype=torch.float16)

                if pos_gen_prompt.dim() == 2:
                    pos_gen_prompt = pos_gen_prompt.unsqueeze(0).expand(n_cls, -1, -1)
                if neg_gen_prompt.dim() == 2:
                    neg_gen_prompt = neg_gen_prompt.unsqueeze(0).expand(n_cls, -1, -1)

                pos_gen_prompts = self.construct_prompts(pos_gen_prompt, pos_prefix, pos_suffix)
                neg_gen_prompts = self.construct_prompts(neg_gen_prompt, neg_prefix, neg_suffix)

                # tokenized_prompts = tokenized_prompts.to(self.device)

                pos_text_features = text_encoder(pos_gen_prompts, pos_tokenized_prompts)
                pos_text_features = pos_text_features / pos_text_features.norm(dim=-1, keepdim=True)

                neg_text_features = text_encoder(neg_gen_prompts, neg_tokenized_prompts)
                neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)

                p_logits = img_i.float() @ pos_text_features.float().t()
                n_logits = img_i.float() @ neg_text_features.float().t()

                l_i = p_logits - cfg.ALPHA * n_logits
                l_i = l_i.squeeze(0)
                logits.append(l_i)
             

            logits = torch.stack(logits)
            
            self.evaluator.process(logits, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    def construct_prompts(self, ctx, prefix, suffix):
        '''
        dim0 is either batch_size (during training) or n_cls (during testing)
        ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        '''
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        
        return prompts

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]
        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)
        return input, label, domain
            

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            # if name == "prompt_learner":
            model_path = os.path.join(directory, name, model_file)

            if not os.path.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = self.load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            optimizer = checkpoint["optimizer"]
            scheduler = checkpoint["scheduler"]
            epoch = checkpoint["epoch"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            # self._models[name].load_state_dict(state_dict, strict=False)
            self._models[name].load_state_dict(state_dict)
            self._optims[name].load_state_dict(optimizer)
            self._scheds[name].load_state_dict(scheduler)

    def load_checkpoint(self, fpath):
        if fpath is None:
            raise ValueError("File path is None")

        if not osp.exists(fpath):
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))

        map_location = "cpu"

        try:
            checkpoint = torch.load(fpath, map_location=map_location)

        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(
                fpath, pickle_module=pickle, map_location=map_location
            )

        except Exception:
            print('Unable to load checkpoint from "{}"'.format(fpath))
            raise

        return checkpoint
    
    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            if self.epoch < self.cfg.EARLY:
                self.before_epoch()
                self.run_epoch()
                self.after_epoch()
            else:
                break
        self.after_train()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % (self.cfg.TRAIN.PRINT_FREQ) == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 
                                if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if (self.epoch + 1) > self.cfg.LATE:
            curr_result = self.test('val')
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.best_epoch = self.epoch

                self.save_model(self.epoch, self.output_dir, model_name="model-best.pth.tar")
                best_val_dir_p = os.path.join(self.output_dir, 'best_val_pos.pt')
                best_val_dir_n = os.path.join(self.output_dir, 'best_val_neg.pt')
                torch.save(self.pmodel, best_val_dir_p)
                torch.save(self.nmodel, best_val_dir_n)

        print('******* Best val acc: {:.1f}%, epoch: {} *******'.format(self.best_result, self.best_epoch+1))

        n_iter = self.epoch
        self.write_scalar("train/val_acc", curr_result, n_iter)
        
        self.set_model_mode("train")
        if self.cfg.SAVE_MODEL and (meet_checkpoint_freq or last_epoch):
            self.save_model(self.epoch, self.output_dir)
    
    
    def after_train(self):
        print("----------Finish training----------")
        print("Deploy the best model")
        model_dir_p = os.path.join(self.output_dir, 'best_val_pos.pt')
        model_dir_n = os.path.join(self.output_dir, 'best_val_neg.pt')
        self.pmodel = torch.load(model_dir_p).to(self.device)
        self.nmodel = torch.load(model_dir_n).to(self.device)
        curr_test_result = self.test('test')
        print('******* Test acc: {:.1f}% *******'.format(curr_test_result))


        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()
