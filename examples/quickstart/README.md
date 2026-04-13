# 快速开始配置说明

docker push ai-Harbor.facethink.com/mlops-public/prepare-agent-v2:04-09-v3

## 可用配置

### CLIP 最小配置

```bash
cp examples/quickstart/clip_minimal.yaml my_config.yaml
```

### Qwen3-VL 最小配置

```bash
cp examples/quickstart/qwen3_vl_minimal.yaml my_config.yaml
```

### 调试配置

```bash
cp examples/quickstart/debug_minimal.yaml my_config.yaml
```

## 推荐流程

### 1. 准备数据

```jsonl
{"query": "a photo of a dog", "positive": "path/to/dog.jpg"}
{"query": "a cat sitting on a chair", "positive": "path/to/cat.jpg"}
```

### 2. 修改关键字段

```yaml
data_path: path/to/my_data.jsonl
image_root: path/to/images
output_dir: output/my_first_run
batch_size: 32
epochs: 3
learning_rate: 5e-5
```

### 3. 启动训练

```bash
vembed train my_config.yaml
```

### 4. 常见覆盖方式

```bash
vembed train examples/quickstart/clip_minimal.yaml     --data_path path/to/my_data.jsonl     --image_root path/to/images     --batch_size 64
```

## 下一步

- 需要更多策略时，查看 `examples/strategies/`
- 需要多 GPU/FSDP 时，查看 `examples/distributed/`
- 需要数据集模板时，查看 `examples/datasets/`
