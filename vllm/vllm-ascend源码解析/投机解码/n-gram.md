# n-gram

模型的输入 (prompt) 与输出之间往往存在大量的 n-gram（连续 token 片段）重合，模型在生成 token 时常常直接从输入中复制它们。每次生成新 token 后，整个上下文序列都会更新，然后用于下一轮的 n-gram 匹配。ngram 本质上是从历史上下文来预测下一个 token，给定当前上下文 token 序列，提取最后的 n 个token，n 的范围由 prompt_lookup_max 和 prompt_lookup_min 决定，在上下文的历史部分中，尝试找到与这些 n 个 token 匹配的模式。

## ngram 推理示例

```
def main():
    prompts = [
        "ABCABCABCABCABCABCABC",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # Create an LLM.
    llm = LLM(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            tensor_parallel_size=1,
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 3, # 每次最多推测 3 个 token
                "prompt_lookup_max": 3, # 最多使用 3 个 n-gram 进行匹配
                "prompt_lookup_min": 2, # 最少使用 2 个 n-gram 进行匹配
            },
            enforce_eager=True,
            gpu_memory_utilization=0.8,
        )  

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
```

根据上面的用例，第一次得到的 prompt_token_ids为 [25411, 25411, 25411, 25411, 25411, 25411, 25411]。在模型执行过程中，首先目标模型会进行一次推理，推理得到 token_id 为 [23354]，然后基于现在请求的所有 token_ids 即 [25411, 25411, 25411, 25411, 25411, 25411, 25411, 23354] 进行 ngram spec decode。
ngram spec decode 步骤：

1. 匹配最大长度的 n-gram (prompt_lookup_max=3)
   首先尝试匹配最后 3 个 token [25411, 25411, 23354],在前面部分[25411, 25411, 25411, 25411, 25411] 找不到对应 [25411, 25411, 23354]的匹配，因此无法匹配
2. 匹配较短的 n-gram (prompt_lookup_min=2)
   接下来尝试匹配最后 2 个 token [25411, 23354],在前面部分[25411, 25411, 25411, 25411, 25411, 25411] 找不到对应 [25411, 23354]的匹配，因此无法匹配
3. 因此生成的draft token为 null