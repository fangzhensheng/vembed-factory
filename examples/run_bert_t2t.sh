#!/bin/bash
# BERT — Text-to-Text retrieval
# Custom model, single encoder for both query and positive
source "$(dirname "$0")/_common.sh"
print_header "BERT — t2t / infonce / lora / gc"
resolve_data true

python run.py examples/bert_t2t.yaml \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH"

echo "Done: experiments/output_bert_t2t/"
