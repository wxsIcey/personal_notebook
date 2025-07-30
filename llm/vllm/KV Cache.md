# 加载模型与预分配显存
vllm在初始化阶段动态计算GPU显存中可分配给KV Cache空间的方法：通过假推理测量显存占用，再反推可用空间。  
KV Cache空间 = GPU总显存 - 非KV Cache的显存占用（包括模型参数，前向推理的中间激活值）
具体步骤为：
- 基于max_num_seqs(假设为3)和max_num_batched_tokens(假设为10)生成假序列[4, 3, 3]
- 使用假数据执行一次前向传播，禁用KV Cache
- 测量此时GPU的显存使用量（即非KV Cache的显存占用）
- KV Cache空间 = GPU总显存 - 非KV Cache的显存占用