import os
import math
import json
import jinja2
from typing import List, Dict, Union

# ==========================================
# 环境变量设置
# ==========================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
hf_token = "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"
os.environ['HF_TOKEN'] = hf_token 

# vLLM 防崩溃保命环境变量
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_ATTENTION_BACKEND'] = 'MATH'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from openai import OpenAI

model_id_list = [
    "google/gemma-7b-it",
    "google/gemma-2-9b-it",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
    "lmsys/vicuna-13b-v1.5",
    "zai-org/glm-4-9b-chat",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2.5-7B-Instruct"
]


plan_test = ["zai-org/GLM-4.7-Flash",
             "meta-llama/Llama-3.3-70B-Instruct",
             "meta-llama/Llama-4-Scout-17B-16E-Instruct"
             "Qwen/Qwen3.5-35B-A3B"
             "Qwen/Qwen3.5-9B"
             
             ]

# ==========================================
# 1. 原始 Transformers 实现 (回滚版本)
# ==========================================
class HuggingFaceModel:
    def __init__(self, repo_name: str, token: str = hf_token):
        """
        使用原生的 transformers 库初始化模型和分词器。
        """
        self.repo_name = repo_name

        # 1. 加载分词器 (Tokenizer)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(repo_name, token=token)
        except (ValueError, OSError):
            self.tokenizer = AutoTokenizer.from_pretrained(repo_name, token=token, trust_remote_code=True)

        if self.tokenizer.chat_template is None:
            print(f"Tokenizer for {repo_name} has no chat template. Applying a default Vicuna template.")
            self.tokenizer.chat_template = (
                "{% if messages[0]['role'] == 'system' %}"
                "{% set loop_messages = messages[1:] %}"
                "{% set system_message = messages[0]['content'] %}"
                "{% else %}"
                "{% set loop_messages = messages %}"
                "{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}"
                "{% endif %}"
                "{{ system_message }}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'] + eos_token }}{% endif %}{% endfor %}"
                "{% if add_generation_prompt and loop_messages[-1]['role'] != 'assistant' %}{{ ' ASSISTANT:' }}{% endif %}"
            )

        # 2. 加载模型 (Model)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                repo_name,
                token=token,
                device_map="auto",
                torch_dtype=torch.bfloat16  
            )
        except (ValueError, OSError):
            self.model = AutoModelForCausalLM.from_pretrained(
                repo_name,
                token=token,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )

        # 3. 确保 pad_token 已设置
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        print(f"Model {repo_name} loaded successfully with transformers.")

    def _prepare_inputs(self, system: str, user: str, Score_wrap: bool = False):
        messages = [
            {'role': 'system', 'content': f'{system}'},
            {'role': 'user', 'content': f'{user}'},
        ]

        if self.repo_name == "Qwen/Qwen3-8B" and Score_wrap == True:
            plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            try:
                plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except jinja2.exceptions.TemplateError:
                messages = [{'role': 'user', 'content': f'{system}\n{user}'}]
                plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(plain_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs, plain_text

    def generate(self, system: str, user: str, max_length: int = 2048, Score_wrap: bool = False, **kwargs):
        inputs, _ = self._prepare_inputs(system, user, Score_wrap)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        response_start = inputs["input_ids"].shape[-1]
        response_ids = outputs[0][response_start:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response

    def conditional_generate(self, condition: str, system: str, user: str, max_length: int = 4096, **kwargs):
        messages = [
            {'role': 'system', 'content': f'{system}'},
            {'role': 'user', 'content': f'{user}'},
        ]
        
        try:
            plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except jinja2.exceptions.TemplateError:
            messages = [{'role': 'user', 'content': f'{system}\n{user}'}]
            plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        plain_text += condition
        inputs = self.tokenizer(plain_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        response_start = inputs["input_ids"].shape[-1]
        response_ids = outputs[0][response_start:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response

    def get_top_k_next_tokens(self, system: str, user: str, k: int = 5, Score_wrap: bool = False) -> List[Dict[str, Union[str, float]]]:
        inputs, plain_text = self._prepare_inputs(system, user, Score_wrap)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        next_token_logits = logits[:, -1, :]
        probabilities = torch.softmax(next_token_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probabilities, k)

        top_k_probs = top_k_probs.cpu().squeeze().tolist()
        top_k_indices = top_k_indices.cpu().squeeze().tolist()

        results = []
        for prob, token_id in zip(top_k_probs, top_k_indices):
            token_str = self.tokenizer.decode(token_id)
            results.append({'token': token_str, 'probability': prob})

        return results


# ==========================================
# 2. 全新独立的 vLLM 实现类
# ==========================================
class VllmModel:
    def __init__(self, repo_name: str, token: str = hf_token, gpu_mem_util: float = 0.9):
        """
        专门使用 vLLM 作为底层的推理模型。
        """
        self.repo_name = repo_name

        print(f"Loading {repo_name} with vLLM Engine...")
        self.model = LLM(
            model=repo_name,
            trust_remote_code=True,
            dtype="bfloat16",       
            tensor_parallel_size=1,        # 默认单卡，避免死锁
            gpu_memory_utilization=gpu_mem_util, 
            enforce_eager=True             # 关 CUDA Graph 防冲突
        )
        
        self.tokenizer = self.model.get_tokenizer()

        if self.tokenizer.chat_template is None:
            print(f"Tokenizer for {repo_name} has no chat template. Applying default.")
            self.tokenizer.chat_template = (
                "{% if messages[0]['role'] == 'system' %}"
                "{% set loop_messages = messages[1:] %}"
                "{% set system_message = messages[0]['content'] %}"
                "{% else %}"
                "{% set loop_messages = messages %}"
                "{% set system_message = 'A chat between a curious user and an artificial intelligence assistant.' %}"
                "{% endif %}"
                "{{ system_message }}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'] + eos_token }}{% endif %}{% endfor %}"
                "{% if add_generation_prompt and loop_messages[-1]['role'] != 'assistant' %}{{ ' ASSISTANT:' }}{% endif %}"
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prepare_inputs(self, system: str, user: str, Score_wrap: bool = False):
        messages = [
            {'role': 'system', 'content': f'{system}'},
            {'role': 'user', 'content': f'{user}'},
        ]

        if self.repo_name == "Qwen/Qwen3-8B" and Score_wrap == True:
            plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            try:
                plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except jinja2.exceptions.TemplateError:
                messages = [{'role': 'user', 'content': f'{system}\n{user}'}]
                plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return None, plain_text

    def generate(self, system: str, user: str, max_length: int = 2048, Score_wrap: bool = False, **kwargs):
        _, plain_text = self._prepare_inputs(system, user, Score_wrap)

        sampling_params = SamplingParams(
            max_tokens=max_length,
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            stop_token_ids=[self.tokenizer.eos_token_id]
        )

        outputs = self.model.generate(
            prompts=[plain_text], 
            sampling_params=sampling_params, 
            use_tqdm=False
        )
        return outputs[0].outputs[0].text

    def conditional_generate(self, condition: str, system: str, user: str, max_length: int = 4096, **kwargs):
        messages = [
            {'role': 'system', 'content': f'{system}'},
            {'role': 'user', 'content': f'{user}'},
        ]
        try:
            plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except jinja2.exceptions.TemplateError:
            messages = [{'role': 'user', 'content': f'{system}\n{user}'}]
            plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        plain_text += condition

        sampling_params = SamplingParams(
            max_tokens=max_length,
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            stop_token_ids=[self.tokenizer.eos_token_id]
        )

        outputs = self.model.generate(
            prompts=[plain_text], 
            sampling_params=sampling_params, 
            use_tqdm=False
        )
        return outputs[0].outputs[0].text

    def get_top_k_next_tokens(self, system: str, user: str, k: int = 5, Score_wrap: bool = False) -> List[Dict[str, Union[str, float]]]:
        _, plain_text = self._prepare_inputs(system, user, Score_wrap)

        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=k
        )

        outputs = self.model.generate(
            prompts=[plain_text], 
            sampling_params=sampling_params, 
            use_tqdm=False
        )

        results = []
        if outputs[0].outputs[0].logprobs:
            top_logprobs_dict = outputs[0].outputs[0].logprobs[0]
            for token_id, logprob_obj in top_logprobs_dict.items():
                prob = math.exp(logprob_obj.logprob)
                token_str = logprob_obj.decoded_token
                if token_str is None:
                    token_str = self.tokenizer.decode(token_id)
                results.append({'token': token_str, 'probability': prob})

        results = sorted(results, key=lambda x: x['probability'], reverse=True)
        return results


# ==========================================
# 3. API 远程调用实现类 (前后端分离专用)
# ==========================================
class ApiModel():
    def __init__(self, model_id, api_key: str = "EMPTY", base_url: str = ""):
        if "gemini" in model_id.lower():
            from google import genai
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id

    def _build_messages(self, system: str, user: str) -> List[Dict[str, str]]:
        """
        辅助拦截器：安全地构建 messages，解决 Gemma / Mistral 等不支持 system 角色的问题
        """
        model_name = self.model_id.lower()
        # 只要是 gemma 或者 mistral，都把 system 暴力合并到 user 提示词前
        if "gemma" in model_name or "mistral" in model_name:
            merged_content = f"{system}\n\n{user}" if system else user
            return [{"role": "user", "content": merged_content}]
        else:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            return messages

    def generate(self, system: str, user_prompt: str, max_length: int = 2000, **kwargs):
        if "gemini" in self.model_id.lower():
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=user_prompt,
            )
            return response.text
        else:
            # 使用安全的拦截器构建 messages
            messages = self._build_messages(system, user_prompt)
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_length,
                temperature=kwargs.get("temperature", 1.0),
                stream=False
            )
            return response.choices[0].message.content

    def conditional_generate(self, condition: str, system: str, user: str, max_length: int = 4096, **kwargs):
        """
        API 模式下的条件生成：通过预填 assistant 角色并让模型续写来实现。
        """
        if "gemini" in self.model_id.lower():
            prompt = f"{user}\n\n[Instruction: Please start your response exactly with '{condition}']"
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
            )
            return condition + " " + response.text
        else:
            # 1. 使用安全的拦截器构建前半部分 messages
            messages = self._build_messages(system, user)
            # 2. 追加 condition 作为 assistant 的续写起点
            messages.append({"role": "assistant", "content": condition})

            try:
                # 尝试高级续写特性
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_length,
                    temperature=kwargs.get("temperature", 1.0),
                    stream=False,
                    extra_body={
                        "continue_final_message": True,
                        "add_generation_prompt": False
                    }
                )
                return condition + response.choices[0].message.content
            except Exception as e:
                print(f"[Warning] continue_final_message fallback triggered for {self.model_id}: {e}")
                # 降级方案：回退到提示词强硬注入
                fallback_prompt = f"{user}\n\nPlease start your response exactly with: {condition}"
                fallback_messages = self._build_messages(system, fallback_prompt)

                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=fallback_messages,
                    max_tokens=max_length,
                    temperature=kwargs.get("temperature", 1.0),
                    stream=False
                )
                return response.choices[0].message.content

    def get_top_k_next_tokens(self, system: str, user: str, k: int = 5, Score_wrap: bool = False) -> List[Dict[str, Union[str, float]]]:
        """
        API 模式下获取 Top-K Token 概率
        """
        import math
        # 使用安全的拦截器构建 messages
        messages = self._build_messages(system, user)
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=k
        )
        
        results = []
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            for logprob_obj in top_logprobs:
                prob = math.exp(logprob_obj.logprob)
                results.append({'token': logprob_obj.token, 'probability': prob})
                
        return sorted(results, key=lambda x: x['probability'], reverse=True)

    def chat(self, messages: List[Dict[str, str]], max_length: int = 8000, **kwargs) -> str:
        """
        Multi-turn chat: send a full conversation history to the model.

        Used by Phase 2 (multi-turn deepening) in the attack loop.
        Handles the same gemma/mistral system-role workaround as _build_messages.

        Args:
            messages: list of {"role": "system"/"user"/"assistant", "content": ...}
                      Full conversation history including system prompt.
            max_length: max tokens to generate.

        Returns:
            The assistant's response text.
        """
        if "gemini" in self.model_id.lower():
            # Gemini: flatten to single prompt
            parts = []
            for msg in messages:
                if msg["role"] == "system":
                    parts.append(f"[System Instructions]: {msg['content']}")
                elif msg["role"] == "user":
                    parts.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    parts.append(f"Assistant: {msg['content']}")
            prompt = "\n\n".join(parts) + "\n\nAssistant:"
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
            )
            return response.text

        # For OpenAI-compatible APIs (vLLM etc.)
        model_name = self.model_id.lower()

        # gemma/mistral don't support system role —
        # merge system message into first user message
        if "gemma" in model_name or "mistral" in model_name:
            processed = []
            system_content = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    processed.append(dict(msg))

            # Prepend system content to the first user message
            if system_content and processed and processed[0]["role"] == "user":
                processed[0]["content"] = f"{system_content}\n\n{processed[0]['content']}"
            elif system_content:
                processed.insert(0, {"role": "user", "content": system_content})

            api_messages = processed
        else:
            api_messages = messages

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=api_messages,
            max_tokens=max_length,
            temperature=kwargs.get("temperature", 0.0),
            stream=False,
        )
        return response.choices[0].message.content or ""