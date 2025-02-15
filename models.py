# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

class Agent:
    """Base agent class for PaSa framework."""
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left'
        )
        
    def infer_score(self, prompts):
        if len(prompts) == 0:
            return []
        encoded_input = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_input.input_ids.cuda(self.model.device)
        attention_mask = encoded_input.attention_mask.cuda(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            output_scores=True, 
            return_dict_in_generate=True, 
            do_sample=False
        )
        true_token_id = self.tokenizer.convert_tokens_to_ids('True')
        probs = outputs.scores[0].softmax(dim=-1)[:, true_token_id].cpu().numpy().tolist()
        return probs

    def infer(self, prompt, sample=False):
        text = self.tokenizer.apply_chat_template(
            [{
                "content": prompt.strip(),
                "role":    "user"
            }],
            tokenize=False,
            max_length=992,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        if sample:
            model_inputs["do_sample"] = True
            model_inputs["temperature"] = 2.0
            model_inputs["top_p"] = 0.8

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def batch_infer(self, prompts, batch_size=8, sample=False):
        if len(prompts) == 0:
            return []
        texts = [self.tokenizer.apply_chat_template(
            [{
                "content": prompt.strip(),
                "role":    "user"
            }],
            tokenize=False,
            max_length=992,
            add_generation_prompt=True
        ) for prompt in prompts]
        responses = []
        for i in range(0, len(texts), batch_size):
            model_inputs = self.tokenizer(texts[i: i + batch_size], return_tensors="pt", truncation=True, padding=True).to(self.model.device)
            if sample:
                model_inputs["do_sample"] = True
                model_inputs["temperature"] = 2.0
                model_inputs["top_p"] = 0.8
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            for response in self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True):
                responses.append(response)
        return responses


class GPT4Agent(Agent):
    """GPT-4 based agent implementation for PaSa-GPT-4o baseline."""
    def __init__(self):
        
        api_key = "sk-XXX"
        base_url = "https://api.chatanywhere.tech/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "gpt-4o-ca"  # Update to "gpt-4o" if available
        self.scoring_cache = {}  # Cache for scoring responses

    def infer_score(self, prompts):
        """Get True/False probabilities using GPT-4 scoring"""
        responses = []
        for prompt in prompts:
            # Cache repeated scoring requests
            if prompt in self.scoring_cache:
                responses.append(self.scoring_cache[prompt])
                continue
                
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user", 
                        "content": prompt
                    }],
                    max_tokens=100,
                    logit_bias={"True": 100, "False": 100},  # Force binary choice
                    temperature=0.0
                )
                
                content = response.choices[0].message.content
                print("Prompt:")
                print(prompt)
                print("Content:")
                print(content)
                score = 1.0 if "True" in content else 0.0
                self.scoring_cache[prompt] = score
                responses.append(score)
            except Exception as e:
                raise RuntimeError(f"GPT-4 API error: {str(e)}")
                
        return responses

    def infer(self, prompt, sample=False):
        """Single inference with GPT-4"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                max_tokens=512,
                temperature=2.0 if sample else 0.7,
                top_p=0.8 if sample else 1.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"GPT-4 API error: {str(e)}")

    def batch_infer(self, prompts, batch_size=8, sample=False):
        """Batch inference using async API calls"""
        from openai import AsyncOpenAI
        import asyncio
        
        async def process_prompt(prompt):
            client = AsyncOpenAI(api_key=self.client.api_key)
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user", 
                        "content": prompt
                    }],
                    max_tokens=512,
                    temperature=2.0 if sample else 0.7,
                    top_p=0.8 if sample else 1.0
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return ""  # Return empty string on failure

        async def run_batch():
            semaphore = asyncio.Semaphore(batch_size)  # Limit concurrent requests
            async with semaphore:
                tasks = [process_prompt(prompt) for prompt in prompts]
                return await asyncio.gather(*tasks)
                
        return asyncio.run(run_batch())
    

    
if __name__ == "__main__":
    selector = Agent("/mnt/hdfs/foundation/agent/heyc/checkpoints/pasa-7b-selector")
    promtp = "You are an elite researcher in the field of AI, conducting research on Give me papers which shows that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets.\n. Evaluate whether the following paper fully satisfies the detailed requirements of the user query and provide your reasoning. Ensure that your decision and reasoning are consistent.\n\nSearched Paper:\nTitle: Specialized Language Models with Cheap Inference from Limited Domain Data\nAbstract:  Abstract Large language models have emerged as a versatile tool but are challenging to apply to tasks lacking large inference budgets and large in-domain training sets. This work formalizes these constraints and distinguishes four important variables: the pretraining budget (for training before the target domain is known), the specialization budget (for training after the target domain is known), the inference budget, and the in-domain training set size. Across these settings, we compare different approaches from the machine learning literature. Limited by inference cost, we find better alternatives to the standard practice of training very large vanilla transformer models. In particular, we show that hyper-networks and mixture of experts have better perplexity for large pretraining budgets, while small models trained on importance sampled datasets are attractive for large specialization budgets. \n\nUser Query: Give me papers which shows that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets.\n\n\nOutput format: Decision: True/False\nReason:... \nDecision:"
    print(selector.infer_score([promtp, promtp, promtp]))
