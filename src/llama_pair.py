"""
Utilities to load a (base, chat) pair of Llama-2 models and generate text
in instruction or chat format. Both tokenizers set pad_token_id = eos_token_id.
Models are torch compiled by default for speed.
"""

from itertools import chain

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):
    """Stop when any of the specified token IDs is generated."""
    def __init__(self, stop_ids):
        super().__init__()
        self.stop_ids = set(stop_ids)

    def __call__(self, input_ids, scores, **kwargs):
        return int(input_ids[0, -1]) in self.stop_ids


class Llama2Agent:
    """
    Loads and holds both:
      1) A base Llama-2 model for instruction–completion.
      2) A chat-tuned Llama-2 model for system/user conversations.
    """

    def __init__(
        self,
        base_model_id: str,
        chat_model_id: str,
        use_4bit: bool = False,
        dtype=torch.float16,
        device_map="auto",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Base model + tokenizer ---
        self.base_tok = AutoTokenizer.from_pretrained(base_model_id)
        # Llama base tokenizer has no pad token; use EOS as pad
        self.base_tok.pad_token_id = self.base_tok.eos_token_id

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            load_in_4bit=use_4bit,
            device_map=device_map,
            #attn_implementation="eager",
        )
        self.base_model = torch.compile(self.base_model)

        # --- Chat model + tokenizer ---
        self.chat_tok = AutoTokenizer.from_pretrained(chat_model_id)
        # Chat tokenizer likewise needs pad token
        self.chat_tok.pad_token_id = self.chat_tok.eos_token_id

        self.chat_model = AutoModelForCausalLM.from_pretrained(
            chat_model_id,
            torch_dtype=dtype,
            load_in_4bit=use_4bit,
            device_map=device_map,
            #attn_implementation="eager"
        )
        self.chat_model = torch.compile(self.chat_model)

    @torch.inference_mode()
    def _batch_generate(
        self,
        model,
        tokenizer,
        prompts,                  # already fully-wrapped strings
        do_sample=False,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        stop_token_ids=None,      # list[int] | None  (same for every prompt)
    ):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        stopping = None
        if stop_token_ids:
            from transformers import StoppingCriteriaList, StoppingCriteria

            class _StopOnAny(StoppingCriteria):
                def __init__(self, ids):
                    super().__init__()
                    self.ids = set(ids)
                def __call__(self, input_ids, scores, **kwargs):
                    return int(input_ids[0, -1]) in self.ids

            stopping = StoppingCriteriaList([_StopOnAny(stop_token_ids)])

        out = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        seq_len = inputs.input_ids.shape[1]
        return [
            tokenizer.decode(out[i, seq_len:], skip_special_tokens=True).strip()
            for i in range(len(prompts))
        ]

    # ----- public helpers that build the correct wrappers -----------
    def generate_base_batch(
        self,
        instructions,                 # list[str]
        stop_sequences=None,
        **gen_kw,
    ):
        template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{inst}\n\n### Response:\n"
        )
        prompts = [template.format(inst=inst) for inst in instructions]

        stop_ids = None
        if stop_sequences:
            stop_ids = sum(
                (self.base_tok(seq, add_special_tokens=False).input_ids
                 for seq in stop_sequences),
                [],
            )

        return self._batch_generate(
            self.base_model,
            self.base_tok,
            prompts,
            stop_token_ids=stop_ids,
            **gen_kw,
        )

    def generate_chat_batch(
        self,
        user_prompts,                # list[str]
        system_prompt="You are a helpful assistant; always answer concisely.",
        **gen_kw,
    ):
        wrap = (
            "<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{usr} [/INST]"
        )
        prompts = [wrap.format(sys=system_prompt, usr=p) for p in user_prompts]

        return self._batch_generate(
            self.chat_model,
            self.chat_tok,
            prompts,
            **gen_kw,
        )

    def generate_base(
        self,
        instruction: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        stop_sequences=None,
        do_sample: bool = True,
    ) -> str:
        """
        Instruction–completion style for the base Llama-2 model.
        """
        # 1) Build Code-Alpaca style prompt
        prompt = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )

        # 2) Tokenize + send to device
        inputs = self.base_tok(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # 3) Prepare stopping criteria if any
        stopping = None
        if stop_sequences:
            stop_ids = list(chain.from_iterable(
                self.base_tok(seq, add_special_tokens=False).input_ids
                for seq in stop_sequences
            ))
            stopping = StoppingCriteriaList([StopOnTokens(stop_ids)])

        # 4) Generate
        with torch.inference_mode():
            out = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.base_tok.pad_token_id,
                eos_token_id=self.base_tok.eos_token_id,
                stopping_criteria=stopping,
            )

        # 5) Decode and strip prompt echo
        gen = out[0, inputs.input_ids.shape[-1] :]
        return self.base_tok.decode(gen, skip_special_tokens=True).strip()


    def generate_chat(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant; always answer concisely.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """
        System+user chat style for the chat-tuned Llama-2 model.
        """
        # 1) Build the standard Llama-2 chat wrapper
        full = (
            "<s>[INST] <<SYS>>\n"
            f"{system_prompt}\n"
            "<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )

        # 2) Tokenize + send to device
        inputs = self.chat_tok(
            full,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # 3) Generate
        with torch.inference_mode():
            out = self.chat_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.chat_tok.pad_token_id,
                eos_token_id=self.chat_tok.eos_token_id,
            )

        # 4) Strip prompt tokens and return
        gen = out[0, inputs.input_ids.shape[-1] :]
        return self.chat_tok.decode(gen, skip_special_tokens=True).strip()
