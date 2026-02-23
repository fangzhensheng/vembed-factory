from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# Define a list of query texts
queries = [
    {"text": "A woman playing with her dog on a beach at sunset."},
    {"text": "Pet owner training dog outdoors near water."},
    {"text": "Woman surfing on waves during a sunny day."},
    {"text": "City skyline view from a high-rise building at night."},
]

# Define a list of document texts and images
documents = [
    {
        "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
    },
    {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    {
        "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    },
]

# Specify the model path
model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"

# Initialize the Qwen3VLEmbedder model
model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)
# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2")

# Combine queries and documents into a single input list
inputs = queries + documents

# Process the inputs to get embeddings
embeddings = model.process(inputs)

# Compute similarity scores between query embeddings and document embeddings
similarity_scores = embeddings[:4] @ embeddings[4:].T

# Print out the similarity scores in a list format
print(similarity_scores.tolist())

# [[0.8157786130905151, 0.7178360223770142, 0.7173429131507874], [0.5195091962814331, 0.3302568793296814, 0.4391537308692932], [0.3884059488773346, 0.285782128572464, 0.33141762018203735], [0.1092604324221611, 0.03871120512485504, 0.06952016055583954]]
