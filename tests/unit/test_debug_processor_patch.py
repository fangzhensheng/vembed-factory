import os
import sys

# Add project root
sys.path.append(os.getcwd())

# Ensure registry is populated
from vembed.model.processors.registry import ProcessorRegistry


def debug_siglip_processor_patch():
    model_name = "google/siglip-base-patch16-224"

    print(f"Resolving processor for {model_name} with mode 'siglip'...")
    try:
        processor = ProcessorRegistry.resolve(model_name, encoder_mode="siglip")
    except Exception as e:
        print(f"Failed to resolve: {e}")
        return

    print(f"Resolved processor type: {type(processor)}")
    print(f"Processor call method: {processor.__call__}")

    texts = ["A cat"]

    # 1. Test default call
    print("\n--- Test 1: Default call ---")
    out = processor(text=texts, return_tensors="pt")
    print(f"Input IDs shape: {out['input_ids'].shape}")

    # 2. Test call with padding=True (simulating collator)
    print("\n--- Test 2: Call with padding=True ---")
    # This is what collator does
    out_dynamic = processor(text=texts, return_tensors="pt", padding=True)
    print(f"Input IDs shape: {out_dynamic['input_ids'].shape}")

    # 3. Test call with explicit max_length (simulating user override)
    print("\n--- Test 3: Call with explicit max_length=32 ---")
    out_explicit = processor(text=texts, return_tensors="pt", padding="max_length", max_length=32)
    print(f"Input IDs shape: {out_explicit['input_ids'].shape}")

    # 4. Check if __call__ is actually patched
    import types

    if isinstance(processor.__call__, types.MethodType):
        print("\n__call__ is a MethodType (Patched)")
    else:
        print("\n__call__ is NOT a MethodType (Not Patched?)")


if __name__ == "__main__":
    debug_siglip_processor_patch()
