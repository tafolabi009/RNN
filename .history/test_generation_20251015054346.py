"""
Quick Test - Generate some text samples
"""

import torch
from inference_improved import ImprovedTextGenerator

# Load model
generator = ImprovedTextGenerator('checkpoints_fast/spectral_ultrafast_best.pth')

print("\n" + "="*80)
print("TESTING TEXT GENERATION")
print("="*80)

# Test prompts
prompts = [
    "The history of artificial intelligence",
    "In the year 2024,",
    "Machine learning is",
    "Once upon a time",
]

settings = [
    {"temp": 0.5, "tokens": 50, "penalty": 1.3},  # Conservative
    {"temp": 0.7, "tokens": 80, "penalty": 1.2},  # Balanced
    {"temp": 0.9, "tokens": 100, "penalty": 1.1}, # Creative
]

for i, prompt in enumerate(prompts):
    setting = settings[i % len(settings)]
    
    print(f"\n{'='*80}")
    print(f"Test {i+1}: {prompt}")
    print(f"Settings: temp={setting['temp']}, tokens={setting['tokens']}, penalty={setting['penalty']}")
    print(f"{'='*80}\n")
    
    text = generator.generate(
        prompt,
        max_new_tokens=setting['tokens'],
        temperature=setting['temp'],
        top_k=40,
        top_p=0.9,
        repetition_penalty=setting['penalty']
    )
    
    print(f"\n📝 Result:")
    print(text)
    print()

print("\n" + "="*80)
print("✅ ALL TESTS COMPLETE")
print("="*80)
