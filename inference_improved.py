"""
IMPROVED INFERENCE - Better Text Generation
============================================

Fixes:
- ✅ Doesn't stop at EOS prematurely
- ✅ Better repetition handling
- ✅ Shows token count and generation stats
- ✅ Handles edge cases
"""

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
import argparse
from pathlib import Path


class ImprovedTextGenerator:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*80}")
        print("LOADING SPECTRAL LANGUAGE MODEL")
        print(f"{'='*80}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        
        # Print info
        print(f"\nModel Configuration:")
        print(f"  Size: {config.hidden_dim}d × {config.num_layers} layers")
        print(f"  Vocabulary: {config.vocab_size:,}")
        print(f"  Max length: {config.max_seq_len}")
        
        num_params = self._count_params(config)
        print(f"  Parameters: ~{num_params/1e6:.1f}M")
        
        if 'epoch' in checkpoint:
            print(f"\nTraining Info:")
            print(f"  Steps: {checkpoint.get('global_step', 'N/A')}")
            print(f"  Epoch: {checkpoint['epoch']}")
            print(f"  Best val loss: {checkpoint.get('best_val_loss', checkpoint.get('val_loss', 'N/A')):.4f}")
        
        # Load model
        from resonance_nn.spectral_optimized import SpectralLanguageModel
        self.model = SpectralLanguageModel(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        print(f"\n📝 Loading tokenizer...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✅ Tokenizer loaded")
        
        print(f"\n{'='*80}")
        print("✅ MODEL READY FOR GENERATION")
        print(f"{'='*80}\n")
    
    def _count_params(self, config):
        """Estimate parameter count"""
        params = (
            config.vocab_size * config.embed_dim +
            config.num_layers * (config.hidden_dim * config.hidden_dim * 8 + config.hidden_dim * 2)
        )
        return params
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,  # Changed from max_length
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        min_length: int = 20,  # New: minimum generation length
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum NEW tokens to generate (not total)
            temperature: Sampling temperature (lower = less random)
            top_k: Keep only top-k tokens
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens (>1 = penalize)
            min_length: Minimum length before allowing EOS
        Returns:
            Generated text
        """
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        prompt_length = input_ids.size(1)
        
        print(f"📊 Input: {prompt_length} tokens")
        
        # Check length
        if prompt_length >= self.model.config.max_seq_len:
            print(f"⚠️  Prompt too long ({prompt_length} tokens), truncating to {self.model.config.max_seq_len - 50}")
            input_ids = input_ids[:, :self.model.config.max_seq_len - 50]
            prompt_length = input_ids.size(1)
        
        # Calculate actual max length
        max_possible = self.model.config.max_seq_len - prompt_length
        tokens_to_generate = min(max_new_tokens, max_possible)
        
        print(f"📊 Will generate: {tokens_to_generate} new tokens (max possible: {max_possible})")
        
        generated = input_ids.clone()
        eos_count = 0  # Track consecutive EOS tokens
        
        # Generate token by token
        for i in range(tokens_to_generate):
            # Forward pass
            logits = self.model(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if token_id < self.model.config.vocab_size:
                        if next_token_logits[0, token_id] < 0:
                            next_token_logits[0, token_id] *= repetition_penalty
                        else:
                            next_token_logits[0, token_id] /= repetition_penalty
            
            # Top-k filtering
            if top_k > 0:
                top_k_clamped = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_clamped)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS (but only after min_length)
            if generated.size(1) - prompt_length >= min_length:
                if next_token.item() == self.tokenizer.eos_token_id:
                    eos_count += 1
                    if eos_count >= 2:  # Two consecutive EOS = stop
                        print(f"📊 Stopped at EOS after {generated.size(1) - prompt_length} tokens")
                        break
                else:
                    eos_count = 0
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        print(f"📊 Generated: {generated.size(1) - prompt_length} new tokens")
        print(f"📊 Total: {generated.size(1)} tokens")
        
        return generated_text
    
    def interactive(self):
        """Interactive mode"""
        print(f"\n{'='*80}")
        print("INTERACTIVE TEXT GENERATION")
        print(f"{'='*80}")
        print("\n💡 Tips for Better Generation:")
        print("  - Lower temperature (0.5-0.7) = more focused")
        print("  - Higher temperature (0.9-1.2) = more creative")
        print("  - Increase repetition_penalty if text repeats")
        print("  - Model works best with 50-150 token generation")
        print("\nCommands:")
        print("  temp X     - Set temperature (e.g., 'temp 0.7')")
        print("  tokens X   - Set max new tokens (e.g., 'tokens 100')")
        print("  topk X     - Set top-k (e.g., 'topk 40')")
        print("  topp X     - Set top-p (e.g., 'topp 0.9')")
        print("  penalty X  - Set repetition penalty (e.g., 'penalty 1.3')")
        print("  quit/exit  - Exit")
        print(f"{'='*80}\n")
        
        # Default settings
        temperature = 0.7
        max_new_tokens = 100
        top_k = 40
        top_p = 0.9
        repetition_penalty = 1.2
        
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
                        print(f"✅ Temperature = {temperature}")
                    except:
                        print("❌ Invalid. Use: temp 0.7")
                    continue
                
                elif prompt.lower().startswith('tokens '):
                    try:
                        max_new_tokens = int(prompt.split()[1])
                        print(f"✅ Max new tokens = {max_new_tokens}")
                    except:
                        print("❌ Invalid. Use: tokens 100")
                    continue
                
                elif prompt.lower().startswith('topk '):
                    try:
                        top_k = int(prompt.split()[1])
                        print(f"✅ Top-k = {top_k}")
                    except:
                        print("❌ Invalid. Use: topk 40")
                    continue
                
                elif prompt.lower().startswith('topp '):
                    try:
                        top_p = float(prompt.split()[1])
                        print(f"✅ Top-p = {top_p}")
                    except:
                        print("❌ Invalid. Use: topp 0.9")
                    continue
                
                elif prompt.lower().startswith('penalty '):
                    try:
                        repetition_penalty = float(prompt.split()[1])
                        print(f"✅ Repetition penalty = {repetition_penalty}")
                    except:
                        print("❌ Invalid. Use: penalty 1.2")
                    continue
                
                # Generate
                print(f"\n🤖 Generating (temp={temperature}, tokens={max_new_tokens}, top_k={top_k}, top_p={top_p}, penalty={repetition_penalty})...")
                print(f"{'='*80}\n")
                
                generated_text = self.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
                
                print(f"\n{'='*80}")
                print("GENERATED TEXT:")
                print(f"{'='*80}")
                print(generated_text)
                print(f"{'='*80}")
            
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Spectral LM Inference (Improved)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = ImprovedTextGenerator(args.checkpoint, args.device)
    
    # Interactive mode
    generator.interactive()


if __name__ == '__main__':
    main()
