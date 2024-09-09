import os
import json
from pathlib import Path

import clip
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from .llama import ModelArgs, Transformer
from .tokenizer import Tokenizer
from .utils import sample_top_p, _download

import time

class LLaMA_adapter(nn.Module):

    def __init__(self, llama_ckpt_dir, llama_tokenizer,
                 max_seq_len=512, max_batch_size=40,
                 clip_model='ViT-L/14',
                 bev_dim=256, query_dim=256,
                 bev_query_len=9, ins_query_len=5,
                 v_embed_dim=768, v_depth=8,
                 v_num_heads=16, v_mlp_ratio=4.0,
                 ins_depth=5, pre_adapter_layer=8,
                 query_len=12, adapter_layer=12,
                 w_bias=False, 
                 w_lora=False, lora_rank=16, 
                 w_new_gate=False,
                 phase="finetune"):
        super().__init__()

        # load llama configs
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        w_bias = phase == "finetune"
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        ) # max_batch_size only affects inference

        # 1. bev projector
        self.bev_dim = bev_dim
        self.downsample8 = nn.Sequential(
            nn.Conv2d(in_channels=self.bev_dim, out_channels=self.bev_dim, kernel_size=6, stride=2, bias=False),
            nn.BatchNorm2d(self.bev_dim), nn.ReLU(),
            nn.Conv2d(in_channels=self.bev_dim, out_channels=self.bev_dim, kernel_size=6, stride=2, bias=False),
            nn.BatchNorm2d(self.bev_dim), nn.ReLU(),
            nn.Conv2d(in_channels=self.bev_dim, out_channels=self.bev_dim, kernel_size=6, stride=2, bias=False),
            nn.BatchNorm2d(self.bev_dim), nn.ReLU(),
        )
        
        self.downsample16 = nn.Sequential(
            # Repeating the structure twice achieves an additional downsampling
            nn.Conv2d(in_channels=self.bev_dim, out_channels=self.bev_dim, kernel_size=6, stride=2, bias=False),
            nn.BatchNorm2d(self.bev_dim), nn.ReLU(),
        )
        
        self.downsample32 = nn.Sequential(
            # Applying downsampling four times in total achieves the desired factor
            nn.Conv2d(in_channels=self.bev_dim, out_channels=self.bev_dim, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(self.bev_dim), nn.ReLU(),
        )

        self.bev_proj = nn.Sequential(
            nn.Linear(self.bev_dim, v_embed_dim),
            nn.LayerNorm(v_embed_dim))

        self.query_len = query_len
        self.bev_query_len = bev_query_len
        self.ins_query_len = ins_query_len
        self.adapter_layer = adapter_layer
        self.query_dim = query_dim
        self.pre_adapter_layer = pre_adapter_layer

        # 2. visual query, blocks and projector
        self.ins_proj = nn.Sequential(
            nn.Linear(query_dim, v_embed_dim),
            nn.LayerNorm(v_embed_dim))
        
        self.sdc_proj = nn.Sequential(
            nn.Linear(query_dim, v_embed_dim),
            nn.LayerNorm(v_embed_dim))
        
        self.planning_proj = nn.Sequential(
            nn.Linear(query_dim, v_embed_dim),
            nn.LayerNorm(v_embed_dim))

        self.bev_blocks = nn.ModuleList([
            Block(v_embed_dim, v_num_heads, v_mlp_ratio, qkv_bias=True)
            for _ in range(v_depth)])
        
        self.ins_blocks = nn.ModuleList([
            Block(v_embed_dim, v_num_heads, v_mlp_ratio, qkv_bias=True)
            for _ in range(ins_depth)])

        self.ctx_proj = nn.Sequential(
            nn.Linear(v_embed_dim, model_args.dim),
            nn.LayerNorm(model_args.dim))

        # 3. learnable query
        # self.bev_prompt_query = nn.Embedding(
        #     bev_query_len, v_embed_dim)

        self.ins_prompt_query = nn.Embedding(
            ins_query_len, v_embed_dim)
        
        self.adapter_query = nn.Embedding(
            query_len * adapter_layer, model_args.dim)
        
        self.pre_adapter_query = nn.Embedding(
            query_len * pre_adapter_layer, model_args.dim)

        # 4. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # 5. llama
        model_args.w_bias = w_bias
        model_args.w_lora = w_lora
        model_args.lora_rank = lora_rank
        model_args.w_new_gate = w_new_gate
        model_args.vocab_size = self.tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt in ckpts:
            print(f"Loading llama ckpt: {ckpt}")
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)

         # 6. training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        # 7. training parameters
        self.phase = phase
        self.get_trainable_params(self.phase)

    def get_trainable_params(self, phase='finetune'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if 'norm' in name or 'bias' in name:
                        para.data = para.data.float()
                        para.requires_grad = True

        elif phase == 'pretrain':
            train_param_name = ['gate', 'downsample', 'bev_proj', 'ins_proj', 'sdc_proj', 'planning_proj', 'bev_blocks', 'ins_blocks', 'ctx_proj', 'adapter_query', 'prompt_query']
            for name, para in self.named_parameters():
                for train_name in train_param_name:
                    if train_name in name:
                        para.data = para.data.float()
                        para.requires_grad = True
        
        else:
            raise ValueError(f"Unknown model phase: {phase}")
        
    def forward_query(self, det_inputs, _bsz): # modified
        feats, ins_query, sdc_query, planning_query = det_inputs

        # bev_prompt_query = self.bev_prompt_query.weight.reshape(1, self.bev_query_len, -1)
        ins_prompt_query = self.ins_prompt_query.weight.reshape(1, self.ins_query_len, -1)
        
        feats = feats.permute(1, 2, 0) # B 256 40000
        bev_size = int(feats.size(-1)**0.5)
        feats = feats.contiguous().view(-1, self.bev_dim, bev_size, bev_size)
        ds8_feats = self.downsample8(feats)  # Downsample by factor of 8
        ds16_feats = self.downsample16(ds8_feats)  # Downsample by factor of 16
        ds32_feats = self.downsample32(ds16_feats)  # Downsample by factor of 32
        
        ds8_feats = self.bev_proj(ds8_feats.view(feats.size(0), self.bev_dim, -1).permute(0, 2, 1))    # B bev_size/8*bev_size/8 v_embed
        ds16_feats = self.bev_proj(ds16_feats.view(feats.size(0), self.bev_dim, -1).permute(0, 2, 1))    # B bev_size/16*bev_size/16 v_embed
        ds32_feats = self.bev_proj(ds32_feats.view(feats.size(0), self.bev_dim, -1).permute(0, 2, 1))    # B bev_size/32*bev_size/32 v_embed
        bev_query = ds32_feats[:, :self.bev_query_len]
        for block in self.bev_blocks:
            bev_query = block(bev_query)

        bev_query = bev_query.repeat((_bsz, 1, 1)) # modi

        if ins_query is not None:
            assert _bsz == len(ins_query)
            ins_query = self.ins_proj(ins_query.float()) # _bsz nq 768
            ins_prompt_query = ins_prompt_query.repeat((_bsz, 1, 1))
        else:
            ins_prompt_query = ins_prompt_query.repeat((_bsz, 1, 1))
            ins_query = torch.zeros_like(ins_prompt_query).to(ins_prompt_query)

        # ins_query = torch.sum(ins_query, dim=1, keepdim=True) # added
        ins_query = torch.cat([ins_prompt_query, ins_query], dim=1)
        for block in self.ins_blocks:
            ins_query= block(ins_query)
        ins_query = ins_query[:, :self.ins_query_len]

        if sdc_query is not None:
            sdc_query = self.sdc_proj(sdc_query.float()).unsqueeze(1)   # B 1 768
        else:
            sdc_query = torch.zeros_like(ins_prompt_query[:1]).to(ins_prompt_query)
        sdc_query = sdc_query.repeat((_bsz, 1, 1))

        if planning_query is not None:
            planning_query = self.planning_proj(planning_query.float()).unsqueeze(1)   # B 1 768
        else:
            planning_query = torch.zeros_like(ins_prompt_query[:1]).to(ins_prompt_query)
        planning_query = planning_query.repeat((_bsz, 1, 1))
        
        ctx_query = torch.cat([bev_query, ins_query, sdc_query, planning_query], dim=1)
        ctx_query = self.ctx_proj(ctx_query)

        return ctx_query

    def forward(self, cap_inputs, det_inputs): # modified
        tokens, labels = cap_inputs
        feats, ins_query, sdc_query, planning_query = det_inputs
        
        _bsz, seqlen = tokens.shape
        ctx_query = self.forward_query(det_inputs, _bsz)
        
        h = self.llama.tok_embeddings(tokens)
        
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        h = self.llama.layers[0](h, 0, freqs_cis, mask, ctx_query)
        
        pre_adapter = self.pre_adapter_query.weight.reshape(self.pre_adapter_layer, self.query_len, -1).unsqueeze(1)
        pre_adapter_index = 0
        for layer in self.llama.layers[1: self.pre_adapter_layer+1]:
            dynamic_adapter = pre_adapter[pre_adapter_index].repeat(_bsz, 1, 1)
            h = layer(h, 0, freqs_cis, mask, dynamic_adapter)
            pre_adapter_index += 1

        for layer in self.llama.layers[self.pre_adapter_layer+1 : -1 * self.adapter_layer]:
            h = layer(h, 0, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.adapter_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.adapter_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            h = layer(h, 0, freqs_cis, mask, dynamic_adapter)
            adapter_index += 1
        
        h = self.llama.norm(h)
        output = self.llama.output(h)
        output = output[:, :-1]
        
        labels = labels[:, 1:]
        tokens = tokens[:, 1:]

        if labels.sum() == 0:
            print("No caption labels")
            c_loss = output.mean() * 0
        else:
            assert self.llama.vocab_size == 32000
            # if torch.isnan(output).any(): # 
            #     fake_loss = torch.zeros(1, requires_grad=True).to(labels.device)
            #     # Use parameters to ensure they participate in the computation graph
            #     for block in self.bev_blocks:
            #         for param in block.parameters():
            #             fake_loss = fake_loss + 0 * param.sum()
            #     print('nan encountered!')
            #     return fake_loss
            assert not torch.isnan(output).any()
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())
            print("c_loss: ", c_loss.item())
        
        # decode ouput and labels
        output = torch.argmax(output, dim=-1)
        decoded = []
        for i, t in enumerate(output.tolist()):
            try:
                label_indices = torch.nonzero(labels[i], as_tuple=True)[0]
                t = [t[idx] for idx in label_indices]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        answer = []
        for i, t in enumerate(tokens.tolist()):
            try:
                label_indices = torch.nonzero(labels[i], as_tuple=True)[0]
                t = [t[idx] for idx in label_indices]
            except ValueError:
                pass
            answer.append(self.tokenizer.decode(t))
        
        print(f"decoded: {decoded}")
        print(f"answer: {answer}")

        return c_loss

    @torch.inference_mode()
    def forward_inference(self, ctx_query, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        h = self.llama.layers[0](h, start_pos, freqs_cis, mask, ctx_query)
        
        pre_adapter = self.pre_adapter_query.weight.reshape(self.pre_adapter_layer, self.query_len, -1).unsqueeze(1)
        pre_adapter_index = 0
        for layer in self.llama.layers[1: self.pre_adapter_layer+1]:
            dynamic_adapter = pre_adapter[pre_adapter_index].repeat(_bsz, 1, 1)
            h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
            pre_adapter_index += 1

        for layer in self.llama.layers[self.pre_adapter_layer+1 : -1 * self.adapter_layer]:
            h = layer(h, start_pos, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.adapter_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.adapter_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float()

    @torch.inference_mode()
    def generate(
        self, det_inputs, prompt_inputs,
        max_gen_len: int = 256,
        temperature: float = 0.2, # default: 0.1
        top_p: float = 0.75,
    ):
        bsz = len(prompt_inputs[0])
        params = self.llama.params

        with torch.cuda.amp.autocast():
            ctx_query = self.forward_query(det_inputs, bsz)
        
        prompts = prompt_inputs[0]
        gt_answers = prompt_inputs[1]
        
        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = t.long()

        input_text_mask = tokens != self.tokenizer.pad_id

        start_pos = min_prompt_size
        prev_pos = 0
        
        check_set = set()
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits = self.forward_inference(ctx_query, tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop
            check_set |= set(i for i in range(bsz) if next_token[i].item() == self.tokenizer.eos_id)
            if len(check_set) == bsz:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                print("No eos token found")
                pass
            decoded.append(self.tokenizer.decode(t))
            
        gt_answer = []
        for i, t in enumerate(gt_answers):
            gt_answer.append(self.tokenizer.decode(t.tolist()))
            
        question = []
        for i, t in enumerate(prompts):
            question.append(self.tokenizer.decode(t.tolist()))
            
        return decoded, gt_answer, question



_MODELS = {
    "BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth",
    "LORA-BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth",
    "CAPTION-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/5088aeb63a89746b90bcfd5cb819e1c7411b2771b267c6d131ce73e250a8abf0_CAPTION-7B.pth",
    "LORA-BIAS-7B-v21": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.1.0/d26d107eec32127ac86ef1997cf7169de1c56a59c539fc1258c6798b969e289c_LORA-BIAS-7B-v21.pth",
    # "LORA16-7B": "",
    # "PARTIAL-7B": ""
}

def available_models():
    return list(_MODELS.keys())

def load(name, llama_dir, llama_type="7B", device="cuda" if torch.cuda.is_available() else "cpu", download_root='ckpts', max_seq_len=512,
        phase="finetune"):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        model_path = name
    else:
        return RuntimeError(f"Model {name} not found; available models = {available_models()}"), None

    # BIAS-7B or https://xxx/sha256_BIAS-7B.pth -> 7B
    # llama_type = name.split('.')[0].split('-')[-1]
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')

    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-Adapter from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')
    model_cfg = ckpt.get('config', {})

    model = LLaMA_adapter(
        llama_ckpt_dir, llama_tokenzier_path,
        max_seq_len=512, max_batch_size=40,
        clip_model='ViT-L/14',
        v_embed_dim=768, v_depth=8,
        v_num_heads=16, v_mlp_ratio=4.0,
        ins_depth=2, pre_adapter_layer=8,
        query_len=12, adapter_layer=12,
        w_bias=model_cfg.get('w_bias', False), 
        w_lora=model_cfg.get('w_lora', False), 
        lora_rank=model_cfg.get('lora_rank', 16),
        w_new_gate=model_cfg.get('w_lora', False), # for compatibility
        phase=phase)

    load_result = model.load_state_dict(ckpt['model'], strict=False)

    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    return model.to(device), model.clip_transform