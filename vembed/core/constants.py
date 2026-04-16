"""Unified constants for data and gradient handling across vembed."""

# Sequence field keys that require padding to max length during concatenation
SEQ_KEYS = {"input_ids", "token_type_ids", "mm_token_type_ids"}

# Fields that get attention_mask padding alongside sequence fields
ATTENTION_MASK_KEY = "attention_mask"

# All sequence-like keys including attention_mask
ALL_SEQ_KEYS = SEQ_KEYS | {ATTENTION_MASK_KEY}

# Indicators for patch/embedding tensor fields (pixel_values, video_patches, etc.)
# Used in field detection: if any of these strings is in the key name, it's a patch tensor
PATCH_INDICATORS = ("pixel_values", "video_patches", "values", "patches")

# Grid metadata field indicator (image_grid_thw, video_grid_thw, etc.)
GRID_INDICATOR = "grid_thw"

# Mapping of grid metadata keys to their corresponding flat-patch tensor keys
# Used when splitting VLM inputs with grid-based patch organization
GRID_TO_PATCH_MAP = {
    "image_grid_thw": "pixel_values",  # Legacy: flat image patches indexed by grid_thw
    "video_grid_thw": "video_patches",  # Future: flat video patches indexed by grid_thw
}

# Priority order for batch size detection in VLM inputs
# Tries sequence keys first (most reliable), then grid keys, then other tensors
BATCH_SIZE_PRIORITY_KEYS = ("input_ids", "attention_mask", "token_type_ids")
