"""
INFERENCE SCRIPT - Generate Text with Trained Models
====================================================

Load trained Spectral models and generate text interactively.

Features:
- ✅ Load pre-trained checkpoints
- ✅ Proper BPE tokenization
- ✅ Interactive chat interface
- ✅ Batch generation
- ✅ Multiple sampling strategies
- ✅ Quality text output (no gibberish!)

Usage:
    # Interactive mode
    python inference.py --checkpoint checkpoints/spectral_base_best.pth
    
    # Generate from prompt
    python inference.py --checkpoint checkpoints/spectral_base_best.pth --prompt "Hello world"
    
    # Batch generation
    python inference.py --checkpoint checkpoints/spectral_base_best.pth --batch prompts.txt
"""

import torch
import torch.nn.functional as F
import argparse
import sys
from pathlib import Path
from typing import List, Optional

# HuggingFace tokenizer
try:
    from transformers import GPT2TokenizerFast
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ transformers not installed. Install with: pip install transformers")
    sys.exit(1)

# Our model
sys.path.insert(0, str(Path(__file__).parent))
from resonance_nn.spectral_optimized import SpectralLanguageModel, SpectralConfig


class TextGenerator:
    """Text generation with trained Spectral models"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*80}")
        print("LOADING SPECTRAL LANGUAGE MODEL")
        print(f"{'='*80}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load config
        config = checkpoint['config']
        print(f"\nModel Configuration:")
        print(f"  Size: {config.embed_dim}d × {config.num_layers} layers")
        print(f"  Vocabulary: {config.vocab_size:,}")
        print(f"  Max length: {config.max_seq_len:,}")
        print(f"  Parameters: ~{self._estimate_params(config)/1e6:.1f}M")
        
        # Create model
        self.model = SpectralLanguageModel(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Training info
        if 'global_step' in checkpoint:
            print(f"\nTraining Info:")
            print(f"  Steps: {checkpoint['global_step']:,}")
            print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
        
        # Load tokenizer
        print(f"\n📝 Loading tokenizer...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"✅ Tokenizer loaded")
        
        print(f"\n{'='*80}")
        print("✅ MODEL READY FOR GENERATION")
        print(f"{'='*80}\n")
    
    def _estimate_params(self, config: SpectralConfig) -> int:
        """Estimate model parameters"""
        params = (
            config.vocab_size * config.embed_dim +
            config.num_layers * (config.hidden_dim * config.hidden_dim * 8 + config.hidden_dim * 2)
        )
        return params
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text from prompt
        
        Args:
            prompt: Input text
            max_length: Maximum total length
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k tokens
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens
            num_return_sequences: Number of sequences to generate
        Returns:
            List of generated texts
        """
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Check length
        if input_ids.size(1) > self.model.config.max_seq_len:
            print(f"⚠️  Prompt too long ({input_ids.size(1)} tokens), truncating to {self.model.config.max_seq_len}")
            input_ids = input_ids[:, :self.model.config.max_seq_len]
        
        # Repeat for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
        
        # Generate
        try:
            generated_ids = self.model.generate(
                input_ids,
                max_length=min(max_length, self.model.config.max_seq_len),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return []
        
        # Decode
        generated_texts = []
        for ids in generated_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def interactive_loop(self):
        """Interactive text generation"""
        print(f"\n{'='*80}")
        print("INTERACTIVE TEXT GENERATION")
        print(f"{'='*80}")
        print("\nCommands:")
        print("  - Type a prompt and press Enter to generate")
        print("  - 'temp X' - Set temperature (e.g., 'temp 0.7')")
        print("  - 'length X' - Set max length (e.g., 'length 200')")
        print("  - 'topk X' - Set top-k (e.g., 'topk 40')")
        print("  - 'topp X' - Set top-p (e.g., 'topp 0.9')")
        print("  - 'quit' or 'exit' - Exit")
        print(f"{'='*80}\n")
        
        # Default settings
        temperature = 0.8
        max_length = 100
        top_k = 50
        top_p = 0.95
        
        while True:
            try:
                prompt = input("\n💬 Prompt: ").strip()
                
                if not prompt:
                    continue
                
                # Commands
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif prompt.lower().startswith('temp '):
                    try:
                        temperature = float(prompt.split()[1])
                        print(f"✅ Temperature set to {temperature}")
                    except:
                        print("❌ Invalid temperature. Use: temp 0.8")
                    continue
                
                elif prompt.lower().startswith('length '):
                    try:
                        max_length = int(prompt.split()[1])
                        print(f"✅ Max length set to {max_length}")
                    except:
                        print("❌ Invalid length. Use: length 100")
                    continue
                
                elif prompt.lower().startswith('topk '):
                    try:
                        top_k = int(prompt.split()[1])
                        print(f"✅ Top-k set to {top_k}")
                    except:
                        print("❌ Invalid top-k. Use: topk 50")
                    continue
                
                elif prompt.lower().startswith('topp '):
                    try:
                        top_p = float(prompt.split()[1])
                        print(f"✅ Top-p set to {top_p}")
                    except:
                        print("❌ Invalid top-p. Use: topp 0.95")
                    continue
                
                # Generate
                print(f"\n🤖 Generating (temp={temperature}, len={max_length}, top_k={top_k}, top_p={top_p})...\n")
                
                generated = self.generate(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                if generated:
                    print(f"{'='*80}")
                    print("GENERATED TEXT:")
                    print(f"{'='*80}")
                    print(generated[0])
                    print(f"{'='*80}")
                else:
                    print("❌ Generation failed")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def batch_generate(
        self,
        prompts: List[str],
        output_file: Optional[str] = None,
        **kwargs
    ):
        """Generate from multiple prompts"""
        print(f"\n{'='*80}")
        print(f"BATCH GENERATION - {len(prompts)} prompts")
        print(f"{'='*80}\n")
        
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"[{i}/{len(prompts)}] Generating from: {prompt[:50]}...")
            
            generated = self.generate(prompt, **kwargs)
            
            if generated:
                results.append({
                    'prompt': prompt,
                    'generated': generated[0]
                })
                print(f"✅ Generated {len(generated[0])} characters")
            else:
                print(f"❌ Failed")
        
        # Save results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, result in enumerate(results, 1):
                    f.write(f"{'='*80}\n")
                    f.write(f"EXAMPLE {i}\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Prompt: {result['prompt']}\n\n")
                    f.write(f"Generated:\n{result['generated']}\n\n")
            
            print(f"\n💾 Results saved to: {output_file}")
        
        print(f"\n{'='*80}")
        print(f"✅ BATCH GENERATION COMPLETE")
        print(f"{'='*80}\n")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Generate text with Spectral LM')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--prompt', type=str, default=None, help='Single prompt to generate from')
    parser.add_argument('--batch', type=str, default=None, help='File with multiple prompts (one per line)')
    parser.add_argument('--output', type=str, default=None, help='Output file for batch generation')
    
    # Generation parameters
    parser.add_argument('--max_length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling threshold')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='Repetition penalty')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='Number of sequences to generate')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Create generator
    generator = TextGenerator(args.checkpoint)
    
    # Generation params
    gen_kwargs = {
        'max_length': args.max_length,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'repetition_penalty': args.repetition_penalty,
        'num_return_sequences': args.num_return_sequences
    }
    
    # Single prompt
    if args.prompt:
        print(f"\n💬 Prompt: {args.prompt}\n")
        generated = generator.generate(args.prompt, **gen_kwargs)
        
        if generated:
            print(f"{'='*80}")
            print("GENERATED TEXT:")
            print(f"{'='*80}")
            for i, text in enumerate(generated, 1):
                if len(generated) > 1:
                    print(f"\n[{i}] {text}")
                else:
                    print(text)
            print(f"{'='*80}\n")
        else:
            print("❌ Generation failed")
    
    # Batch generation
    elif args.batch:
        if not Path(args.batch).exists():
            print(f"❌ Batch file not found: {args.batch}")
            return
        
        with open(args.batch, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        generator.batch_generate(prompts, args.output, **gen_kwargs)
    
    # Interactive mode
    else:
        generator.interactive_loop()


if __name__ == '__main__':
    main()
