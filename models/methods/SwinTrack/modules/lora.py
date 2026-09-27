"""LoRA 低秩旁路 Linear 包装器。

用于在冻结的 nn.Linear 基础上叠加一个低秩可训练旁路, 通过 active 开关按模态
选择性生效 (例如只在处理 TIR 流时打开)。weight/bias 与原 nn.Linear 保持同名
顶层 Parameter, 因此 state_dict key 不变, 可直接热启动已训练好的 checkpoint
(旧 key 精确命中原始权重, 新增的 lora_A/lora_B 保持各自初始化)。
"""
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """frozen base Linear + 低秩旁路 (lora_B 零初始化, 初始严格等价原 Linear)。"""

    def __init__(self, base_linear: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        nn.init.constant_(self.lora_B.weight, 0.)
        self.scale = alpha / r

        # 外部按当前处理的模态切换: True=叠加 LoRA 旁路, False=纯 frozen base
        self.active = False

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.active:
            out = out + self.scale * self.lora_B(self.lora_A(x))
        return out
