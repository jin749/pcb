import os.path as osp
from random import sample 
import time 
import json 

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.datasets import build_dataset
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .active_learning.pcb import PCB
from .active_learning.badge import BADGE
from .active_learning.coreset import Coreset
from .active_learning.entropy import Entropy
from .active_learning.warmstart import WarmStart
from .active_learning.badge_for_filter import BADGE as F_BADGE
from .active_learning.coreset_for_filter import Coreset as F_Coreset

import wandb
import contextlib
import sys
import os

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

    model = clip.build_model(state_dict or model.state_dict())
    
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

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # if not ctx_init.endswith(".json"):
        prompt_prefix = " ".join(["X"] * n_ctx)
        
        classnames = [name.replace("_", " ") for name in classnames]
        if cfg.TRAINER.COOPAL.ASPATH:
            with open(f"descriptors/descriptors_{cfg.TRAINER.COOPAL.ASPATH}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
                
            name_lens, prompts = [], []
            for name in classnames:
                name = name.lower()
                for desc in desc_dict[name]:
                    name_lens.append(len(_tokenizer.encode(f"{name}, which is/has {desc}")))
                    prompts.append(prompt_prefix + " " + f"{name}, which is/has {desc}.")
                    
        elif cfg.TRAINER.COOPAL.AEPATH:
            with open(f"descriptors/descriptors_{cfg.TRAINER.COOPAL.AEPATH}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
                
            name_lens, prompts = [], []
            for name in classnames:
                name = name.lower()
                for desc in desc_dict[name]:
                    name_lens.append(len(_tokenizer.encode(f"{name}, which is/has {desc}")))
                    prompts.append(prompt_prefix + " " + f"{name}, which is/has {desc}.")
                    
        else:
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.prompts = prompts
        print(prompts[:10])
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
       
       
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = embedding.size(0)
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
       
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(self.n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

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
    

class WeightedSumOfLogits(nn.Module):
    def __init__(self, n_class_desc, dtype, weighted_sum_weight=None):
        super().__init__()
        self.n_class_desc = n_class_desc
        
        if weighted_sum_weight is None:
            initial_weighted_sum_weight = []
            for n in self.n_class_desc:
                initial_weighted_sum_weight.extend([1/n] * n)
            self.w = nn.Parameter(torch.tensor(initial_weighted_sum_weight, dtype=dtype))

        else:
            self.w = nn.Parameter(weighted_sum_weight)
            
    def forward(self, augmented_logits):
        weighted_sum_logits = []
        start = 0
        for n in self.n_class_desc:
            same_class_logits = augmented_logits[:, start: start + n]
            same_class_softmaxed_weight = F.softmax(self.w[start: start + n], dim=0)
            weighted_sum_logit = same_class_logits @ same_class_softmaxed_weight

            weighted_sum_logits.append(weighted_sum_logit)
            start += n
        
        weighted_sum_logits = torch.stack(weighted_sum_logits, dim=1)
        
        return weighted_sum_logits
    


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, desc_file=None, weighted_sum_weight=None):
        super().__init__()
        
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_class_desc=[]
        self.n_cls = len(classnames)
        self.cfg = cfg

        if desc_file is not None:
            with open(f"descriptors/descriptors_{desc_file}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
            classnames = [name.replace("_", " ") for name in classnames]
            for name in classnames:
                name = name.lower()
                self.n_class_desc.append(len(desc_dict[name]))

        self.weighted_sum_weight = weighted_sum_weight
        self.weighted_sum = WeightedSumOfLogits(self.n_class_desc, self.dtype, self.weighted_sum_weight)
        
            
        
    def forward(self, image, get_feature=False):
        image_features = self.image_encoder(image.type(self.dtype))
        
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
    
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        if self.cfg.TRAINER.COOPAL.AEPATH:
            tmp = []
            start = 0
            for n in self.n_class_desc:
                tmp.append(text_features[start:start+n].mean(dim=0))
                start += n
            text_features = torch.stack(tmp)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        if self.cfg.TRAINER.COOPAL.ASPATH: 
            if self.cfg.TRAINER.COOPAL.FILTER: # TODO: filter
                logits = self.weighted_sum(logits)
            else:
                tmp = [] 
                start = 0
                for n in self.n_class_desc:
                    tmp.append(torch.sum(logits[:, start:start+n], dim=1)/n)
                    start += n
                logits = torch.stack(tmp, dim=1)

        if get_feature:
            return logits, image_features
        else:
            return logits


@TRAINER_REGISTRY.register()
class ALVLM(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.acc = []
        
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self, weighted_sum_weight=None):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        if cfg.TRAINER.COOPAL.ASPATH:
            self.model = CustomCLIP(cfg, classnames, clip_model, desc_file=cfg.TRAINER.COOPAL.ASPATH, weighted_sum_weight=weighted_sum_weight)
        elif cfg.TRAINER.COOPAL.AEPATH:
            self.model = CustomCLIP(cfg, classnames, clip_model, desc_file=cfg.TRAINER.COOPAL.AEPATH)
        else:
            self.model = CustomCLIP(cfg, classnames, clip_model)
        #print(self.model)
        
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "weighted_sum" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(f"prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        filter_optim_cfg = get_filter_optim_cfg(cfg)
        self.filter_optim = build_optimizer(self.model.weighted_sum, filter_optim_cfg)
        self.filter_sched = build_lr_scheduler(self.filter_optim, filter_optim_cfg)
        self.register_model("weighted_sum", self.model.weighted_sum, self.filter_optim, self.filter_sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            #print(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

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
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    def before_train(self, weighted_sum_weight=None):
        print("INITIALIZE the prompts weights")
        self.build_model(weighted_sum_weight)
        
    def after_train(self):
        print("Finish training")
        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.acc.append(self.test())
            
        # Close writer
        self.close_writer()
        
    def train(self):
        MODE = None
        TARGET_ROUND = 8
        if self.cfg.TRAINER.COOPAL.AEPATH:
            MODE = "AE"
        elif self.cfg.TRAINER.COOPAL.ASPATH:
            MODE = "AS"
        else:
            MODE = "none"
        if self.cfg.WANDB_PROJECT_NAME:
            os.environ["WANDB__SERVICE_WAIT"] = "300"
            wandb.init(
                project = self.cfg.WANDB_PROJECT_NAME,
                entity=self.cfg.WANDB_ENTITY,
                group = os.path.join(*self.cfg.OUTPUT_DIR.split('/')[1:-1]),
                name = self.cfg.OUTPUT_DIR,
                config={
                    "DATASET": self.cfg.DATASET.NAME,
                    "TRAINER": self.cfg.TRAINER.NAME,
                    "BACKBONE": self.cfg.MODEL.BACKBONE.NAME,
                    "ALMETHOD": self.cfg.TRAINER.COOPAL.METHOD,
                    "MODE": MODE,
                    "WARM_START": self.cfg.TRAINER.COOPAL.WARM_START,
                    "FILTER": self.cfg.TRAINER.COOPAL.FILTER,
                    "FILTER_OPTIM_NAME": self.cfg.TRAINER.COOPAL.FILTER_OPTIM_NAME,
                    "FILTER_LR": self.cfg.TRAINER.COOPAL.FILTER_LR,
                    "ALMETHOD_FOR_FILTER": self.cfg.TRAINER.COOPAL.ALMETHOD_FOR_FILTER,
                    "SEED": self.cfg.SEED,
                    "NUM_SHOTS": 1,
                    "TARGET_ROUND": TARGET_ROUND,
                    "CSC": self.cfg.TRAINER.COOP.CSC,
                }
            )

        """Generic training loops."""
        dataset = build_dataset(self.cfg)
        
        print(f"dataset length: {len(dataset.train_x)}")
        unlabeled_dst = dataset.train_x 
        U_index = list(range(len(unlabeled_dst)))
        if self.cfg.TRAINER.COOP.CSC:
            n_query = dataset.get_num_classes(unlabeled_dst)
        else:
            n_query = dataset.get_num_classes(unlabeled_dst)
        n_cand = int(len(unlabeled_dst) * self.cfg.TRAINER.COOPAL.GAMMA) # 10% of entire dataset
        
        dataset._train_x = []
        weighted_sum_weight = None
        for i in range(TARGET_ROUND):
            start = time.time()
            if self.cfg.TRAINER.COOPAL.METHOD == "random" or (self.cfg.TRAINER.COOPAL.WARM_START == False) and i == 0:
                idx = sample(U_index, n_query)
                
            elif self.cfg.TRAINER.COOPAL.WARM_START == True and i == 0:
                self.before_train()
                print("\n\n Warm Start \n\n")
                selector = WarmStart(self.cfg, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), self.device)
                idx = selector.select(n_query)

            elif self.cfg.TRAINER.COOPAL.METHOD == "entropy":
                selector = Entropy(self.cfg, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), self.device)
                idx = selector.select(n_cand)

            elif self.cfg.TRAINER.COOPAL.METHOD == "badge":
                if self.cfg.TRAINER.COOPAL.ALMETHOD_FOR_FILTER:
                    print("\n\n BADGE FOR FILTER \n\n")
                    selector = F_BADGE(self.cfg, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), self.device)
                else:
                    selector = BADGE(self.cfg, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), self.device)
                idx = selector.select(n_cand)

            elif self.cfg.TRAINER.COOPAL.METHOD == "coreset":
                val_x = dataset._train_x.copy()
                if self.cfg.TRAINER.COOPAL.ALMETHOD_FOR_FILTER:
                    print("\n\n CORESET FOR FILTER \n\n")
                    selector = F_Coreset(self.cfg, self.model, unlabeled_dst, U_index, val_x, dataset.get_num_classes(unlabeled_dst))
                else:
                    selector = Coreset(self.cfg, self.model, unlabeled_dst, U_index, val_x, dataset.get_num_classes(unlabeled_dst))
                idx = selector.select(n_cand)

            else:
                print("NotImplementedError")
                idx = U_index
            
            if i != 0:
                statistics = torch.zeros(self.num_classes)
                for elem in dataset._train_x:
                    statistics[elem.label] += 1
                selector = PCB(self.cfg, self.model, unlabeled_dst, idx, dataset.get_num_classes(unlabeled_dst), statistics, self.device)
                idx = selector.select(n_query)
            
            for k in idx:
                dataset._train_x.append(unlabeled_dst[k])
                U_index.remove(k)
            assert len(U_index) + len(dataset.train_x) == len(unlabeled_dst), f"u index: {len(U_index)}\t train set: {len(dataset.train_x)}\t unlabeled_dst: {len(unlabeled_dst)}"
            
            self.train_loader_x = build_data_loader(
                self.cfg,
                sampler_type=self.cfg.DATALOADER.TRAIN_X.SAMPLER,
                data_source=dataset.train_x,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=True),
                is_train=True,
                dataset_wrapper=None
            )   

            self.before_train(weighted_sum_weight) # build model

            print("\n\n=== Start training ===")
            if torch.cuda.device_count() > 1:
                model_for_eval = self.model.module
            else:
                model_for_eval = self.model

            print_filter(model_for_eval)
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.run_epoch()
                self.after_epoch()
            self.after_train()

            print("\n\n=== End training ===")
            print_filter(model_for_eval)
            weighted_sum_weight = model_for_eval.weighted_sum.w.detach()

            print("\nTraining time for {}-th round: {:.2f} seconds".format(i, time.time() - start))

            if self.cfg.WANDB_PROJECT_NAME:
                wandb.log({"acc": self.acc[-1], "round": i+1, "total_budget": n_query*(i+1), "round_budget": n_query}, step=n_query*(i+1))

        print("=== Result Overview ===")
        for i in range(len(self.acc)):
            print(f"{i}: {self.acc[i]}")
        print("=======================")    


def get_filter_optim_cfg(cfg):
    from yacs.config import CfgNode as CN

    filter_optim = CN(cfg.OPTIM)
    filter_optim.NAME = cfg.TRAINER.COOPAL.FILTER_OPTIM_NAME

    if cfg.TRAINER.COOPAL.FILTER_LR:
        filter_optim.LR = cfg.TRAINER.COOPAL.FILTER_LR

    for k, v in cfg.OPTIM.items():
        if k not in filter_optim.keys():
            filter_optim[k] = v

    print("\n\nFilter Optimizer Config: ")
    print(filter_optim)

    return filter_optim


def print_filter(model,num_of_displayed_class=10):
    n_class_desc = model.n_class_desc
    prompts = model.prompt_learner.prompts
    weighted_sum_weight = model.weighted_sum.w.detach()

    start = 0
    softmaxed_weight = []
    start = 0
    for n in n_class_desc:
        same_class_softmaxed_weight = torch.nn.functional.softmax(weighted_sum_weight[start: start + n], dim=0)
        softmaxed_weight.extend(same_class_softmaxed_weight)
        start += n
    
    print("\n\n <Filtered descriptions>")
    print("[weight]\t[softmaxed weight] * [description]")
    start = 0
    displayed_class = 0
    for n in n_class_desc:
        for i in range(n):
            print(f"{weighted_sum_weight[start + i].item():.4f}\t\t{softmaxed_weight[start + i].tolist():.4f} * \"{prompts[start + i]}\"")
        start += n
        displayed_class += 1
        if displayed_class >= num_of_displayed_class:
            break
