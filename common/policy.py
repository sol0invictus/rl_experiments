"""LoRA policy shared by every experiment: load the model, one policy-gradient step on captured tokens, and
push the updated policy to vLLM (merged weights or a LoRA adapter).

An experiment's train.py builds its own sampling loop and turns rollouts into (row, advantage) samples; a row
is {"ids", "mask", "old_lp"} as produced from the capture proxy's calls. Everything below is experiment-agnostic.
"""
import asyncio
import importlib.util
import os
from collections import defaultdict

import aiohttp
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer


def add_args(p):
    """CLI options every experiment's train.py shares (experiment-specific ones are added by the caller)."""
    p.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
    p.add_argument("--tasks-per-step", type=int, default=8)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--max-pack", type=int, default=8192)
    p.add_argument("--max-tokens", type=int, default=4096, help="output cap per model call (blog: 4096)")
    p.add_argument("--max-ctx", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--eval-temperature", type=float, default=0.7)
    p.add_argument("--is-cap", type=float, default=2.0, help="truncated importance-sampling cap")
    p.add_argument("--adv-std-floor", type=float, default=0.05)
    p.add_argument("--vllm", default="http://127.0.0.1:8000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-step", type=int, default=0,
                   help="continue a crashed run from adapters/step_N (optimizer state too if optimizer.pt exists)")
    p.add_argument("--sync", choices=["lora", "merge"], default="merge",
                   help="push the policy to vLLM as a LoRA adapter or as merged full weights")
    return p


class Policy:
    def __init__(self, args, run_dir):
        self.args, self.run_dir = args, run_dir
        self.tok = AutoTokenizer.from_pretrained(args.model)
        attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                     attn_implementation=attn).cuda()
        # LFM2's short convolutions would leak state across rows packed into one sequence, so pack only
        # pure-attention models; LFM2 trains one row per forward.
        self.pack = attn == "flash_attention_2" and "conv" not in (getattr(model.config, "layer_types", None) or [])
        if model.config.model_type == "lfm2":    # attention + MLP (vLLM serves these as qkv_proj / w13 / w2)
            targets = r".*\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|feed_forward\.(w1|w2|w3))"
        else:
            targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        self.model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=targets))
        self.model.print_trainable_parameters()
        self.opt = torch.optim.AdamW([q for q in self.model.parameters() if q.requires_grad], lr=args.lr,
                                     betas=(0.9, 0.99), weight_decay=0.0)
        self.backbone = self.model.base_model.model.model          # LoRA layers are injected in place
        self.lm_head_w = self.model.base_model.model.lm_head.weight

    # ------------------------------------------------------------------------------------ training
    def _chunk_logprobs(self, h, targets):
        logits = (h @ self.lm_head_w.T).float()
        return logits.log_softmax(-1).gather(-1, targets[:, None])[:, 0]

    def packed_logprobs(self, rows):
        """Log-probs of every loss token in a pack of rows (flash-attn varlen via position_ids resets)."""
        ids = torch.tensor([t for r in rows for t in r["ids"]], device="cuda")[None]
        pos = torch.tensor([i for r in rows for i in range(len(r["ids"]))], device="cuda")[None]
        hidden = self.backbone(input_ids=ids, position_ids=pos).last_hidden_state[0]
        # predict token t from position t-1; loss positions are where mask==1 (never the first token of a row)
        sel, off = [], 0
        for r in rows:
            sel += [off + i - 1 for i, m in enumerate(r["mask"]) if m]
            off += len(r["ids"])
        sel = torch.tensor(sel, device="cuda")
        h, tgt = hidden[sel], ids[0, sel + 1]
        out = [checkpoint(self._chunk_logprobs, h[i:i + 2048], tgt[i:i + 2048], use_reentrant=False)
               for i in range(0, len(sel), 2048)]
        return torch.cat(out)

    def train_step(self, samples):
        """samples: list of (row, advantage). One optimizer step, token-level mean over the batch."""
        args, model = self.args, self.model
        model.train()
        n_tok = sum(sum(r["mask"]) for r, _ in samples)
        packs, cur, cur_len = [], [], 0
        for s in sorted(samples, key=lambda s: -len(s[0]["ids"])):
            L = len(s[0]["ids"])
            if cur and (not self.pack or cur_len + L > args.max_pack):
                packs.append(cur)
                cur, cur_len = [], 0
            cur.append(s)
            cur_len += L
        if cur:
            packs.append(cur)
        stats = defaultdict(float)
        for pack in packs:
            rows = [r for r, _ in pack]
            lp = self.packed_logprobs(rows)
            old = torch.tensor([x for r in rows for x, m in zip(r["old_lp"], r["mask"]) if m], device="cuda")
            adv = torch.tensor([a for r, a in pack for m in r["mask"] if m], device="cuda")
            log_ratio = lp.detach() - old                     # trainer vs sampler (vLLM) mismatch
            w = log_ratio.exp().clamp(max=args.is_cap)
            loss = -(w * adv * lp).sum() / n_tok
            loss.backward()
            stats["loss"] += loss.item()
            stats["mismatch_kl"] += (-log_ratio).sum().item() / n_tok   # E_sampler[log q - log p]
            stats["is_clipped"] += (log_ratio.exp() > args.is_cap).sum().item() / n_tok
            stats["entropy_proxy"] += (-lp.detach()).sum().item() / n_tok
        gn = torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        stats.update(grad_norm=gn.item(), train_tokens=n_tok, packs=len(packs),
                     peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                     seq_tokens=sum(len(r["ids"]) for r, _ in samples))
        return dict(stats)

    # ------------------------------------------------------------------------- vLLM weight sync
    async def publish_adapter(self, session, step, prev_name):
        args = self.args
        path = os.path.join(self.run_dir, "adapters", f"step_{step}")
        self.model.save_pretrained(path)
        name = f"{args.run}_s{step}"
        for attempt in range(5):   # vLLM can stall on the load while it drains requests; don't kill the run
            try:
                body = {"lora_name": name, "lora_path": path, "load_inplace": True}   # idempotent across retries
                async with session.post(f"{args.vllm}/v1/load_lora_adapter", json=body,
                                        timeout=aiohttp.ClientTimeout(total=600)) as r:
                    assert r.status == 200, await r.text()
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"load_lora_adapter attempt {attempt + 1} failed: {e!r}", flush=True)
                if attempt == 4:
                    raise
                await asyncio.sleep(30)
        if prev_name != args.model:
            async with session.post(f"{args.vllm}/v1/unload_lora_adapter", json={"lora_name": prev_name}) as r:
                pass
        return name

    @torch.no_grad()
    def merged_state_dict(self):
        """Base weights with W + scale * B @ A folded in (fp32 math, base weights untouched), in HF names."""
        model, sd = self.model, {}
        for name, mod in model.base_model.model.named_modules():
            if hasattr(mod, "lora_A") and "default" in mod.lora_A:
                delta = mod.lora_B["default"].weight.float() @ mod.lora_A["default"].weight.float()
                w = mod.base_layer.weight.float() + mod.scaling["default"] * delta
                sd[name + ".weight"] = w.to(torch.bfloat16).cpu()
        for name, prm in model.base_model.model.named_parameters():
            name = name.replace(".base_layer", "")
            if "lora_" in name or name in sd or (name == "lm_head.weight" and model.config.tie_word_embeddings):
                continue
            sd[name] = prm.detach().cpu()
        return sd

    async def publish_merged(self, session, step):
        """Sync the policy into vLLM by reloading full merged weights in place. vLLM's LoRA path drops out of
        CUDA graphs for LFM2 and decodes ~15x slower on this box, so the server runs without LoRA."""
        args = self.args
        if step > 0:
            self.model.save_pretrained(os.path.join(self.run_dir, "adapters", f"step_{step}"))   # small checkpoint
            torch.save(self.opt.state_dict(), os.path.join(self.run_dir, "optimizer.pt"))  # latest only, for --resume-step
        path = os.path.join(self.run_dir, "policy")
        os.makedirs(path, exist_ok=True)
        save_file(self.merged_state_dict(), os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
        rpc = {"method": "reload_weights", "kwargs": {"weights_path": path}}
        for attempt in range(5):   # one connect stall killed a run at step 21; reloading is idempotent, so retry
            try:
                async with session.post(f"{args.vllm}/collective_rpc", json=rpc,
                                        timeout=aiohttp.ClientTimeout(total=600, sock_connect=60)) as r:
                    assert r.status == 200, await r.text()
                async with session.post(f"{args.vllm}/reset_prefix_cache",   # cached KV came from the old weights
                                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=60)) as r:
                    assert r.status == 200, await r.text()
                return
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"weight sync attempt {attempt + 1} failed: {e!r}", flush=True)
                if attempt == 4:
                    raise
                await asyncio.sleep(30)

    async def sync(self, session, step, proxy):
        """Push the current policy to vLLM with the configured --sync mode."""
        if self.args.sync == "merge":
            await self.publish_merged(session, step)
        else:
            proxy.model_name = await self.publish_adapter(session, step, proxy.model_name)

    async def resume(self, session, proxy):
        """Load adapters/step_N (and optimizer.pt if present) for --resume-step N, then sync vLLM."""
        step = self.args.resume_step
        adir = os.path.join(self.run_dir, "adapters", f"step_{step}")
        res = set_peft_model_state_dict(self.model, load_file(os.path.join(adir, "adapter_model.safetensors"),
                                                              device="cuda"))
        assert not res.unexpected_keys, res.unexpected_keys[:5]
        opt_path = os.path.join(self.run_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            self.opt.load_state_dict(torch.load(opt_path))
        else:
            print("resume: no optimizer.pt, AdamW moments restart from zero", flush=True)
        if self.args.sync == "merge":
            await self.publish_merged(session, 0)
        else:
            proxy.model_name = await self.publish_adapter(session, step, proxy.model_name)
