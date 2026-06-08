## 模型训练 & 测试 infra

### 安装

```bash
pip install -r requirements.txt
```

### SFT 训练

```bash
# 用 YAML 配置文件
python sft.py --config configs/sft_qwen2.5_1.5b.yaml

# 或用命令行参数（覆盖 YAML）
python sft.py \
  --model ./Qwen2.5-1.5B \
  --data ./data/sft_chat_22k.jsonl \
  --output ./output/experiment \
  --epochs 5 --batch_size 4
```

### 数据格式

JSONL，每行一条，`messages` 字段：

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
