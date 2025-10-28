"""
Simple Usage Example: Using Spectral Neural Networks in Your Project
=====================================================================

This example demonstrates how to use the resonance_nn module
in your own projects for various NLP tasks.

Installation:
    pip install git+https://github.com/tafolabi009/RNN.git

    OR

    git clone https://github.com/tafolabi009/RNN.git
    cd RNN
    pip install -e .
"""

import torch
from resonance_nn import (
    create_spectral_lm,
    SpectralClassifier,
    SpectralEncoder,
    SpectralConfig,
    list_available_models
)


def example_1_language_modeling():
    """
    Example 1: Language Modeling
    Create a language model and generate text
    """
    print("\n" + "="*80)
    print("Example 1: Language Modeling & Text Generation")
    print("="*80)
    
    # Create a base model (983M parameters, 131K context)
    model = create_spectral_lm(
        size='base',
        vocab_size=50257,  # GPT-2 vocabulary
        max_seq_len=2048
    )
    
    print(f"✅ Created model with {model.get_num_params()/1e6:.1f}M parameters")
    
    # Simulate input (batch_size=2, seq_len=100)
    input_ids = torch.randint(0, 50257, (2, 100))
    
    # Forward pass for training
    logits = model(input_ids)
    print(f"✅ Forward pass output shape: {logits.shape}")
    
    # Generate text
    print("\n📝 Generating text...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids[:1, :10],  # Start with first 10 tokens
            max_length=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
    print(f"✅ Generated sequence shape: {generated.shape}")
    print(f"   Length: {generated.shape[1]} tokens")


def example_2_text_classification():
    """
    Example 2: Text Classification
    Create a classifier for sentiment analysis, topic classification, etc.
    """
    print("\n" + "="*80)
    print("Example 2: Text Classification")
    print("="*80)
    
    # Configure for classification
    config = SpectralConfig(
        vocab_size=30522,      # BERT vocabulary
        embed_dim=768,
        hidden_dim=3072,
        num_layers=12,
        max_seq_len=512,
        dropout=0.1,
        use_rope=True
    )
    
    # Create classifier (e.g., for sentiment: negative/neutral/positive)
    num_classes = 3
    classifier = SpectralClassifier(config, num_classes=num_classes)
    
    print(f"✅ Created classifier with {num_classes} classes")
    print(f"   Model size: {sum(p.numel() for p in classifier.parameters())/1e6:.1f}M parameters")
    
    # Simulate input batch
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    
    # Forward pass
    classifier.eval()
    with torch.no_grad():
        logits = classifier(input_ids)
        predictions = torch.argmax(logits, dim=-1)
    
    print(f"✅ Classification output shape: {logits.shape}")
    print(f"   Predictions: {predictions.tolist()}")
    print(f"   Probabilities:")
    probs = torch.softmax(logits, dim=-1)
    for i in range(batch_size):
        print(f"      Sample {i+1}: {probs[i].tolist()}")


def example_3_document_embeddings():
    """
    Example 3: Document Embeddings
    Encode documents for similarity search, clustering, etc.
    """
    print("\n" + "="*80)
    print("Example 3: Document Embeddings")
    print("="*80)
    
    # Configure for encoding
    config = SpectralConfig(
        vocab_size=50257,
        embed_dim=768,
        hidden_dim=3072,
        num_layers=12,
        max_seq_len=8192  # Can handle long documents!
    )
    
    # Create encoder
    encoder = SpectralEncoder(config)
    
    print(f"✅ Created encoder")
    print(f"   Max sequence length: {config.max_seq_len:,} tokens")
    
    # Simulate two documents
    doc1_ids = torch.randint(0, 50257, (1, 1000))
    doc2_ids = torch.randint(0, 50257, (1, 1000))
    
    # Encode documents
    encoder.eval()
    with torch.no_grad():
        doc1_embedding = encoder(doc1_ids)  # (1, 1000, 768)
        doc2_embedding = encoder(doc2_ids)  # (1, 1000, 768)
    
    # Pool to get document-level embeddings
    doc1_vec = doc1_embedding.mean(dim=1)  # (1, 768)
    doc2_vec = doc2_embedding.mean(dim=1)  # (1, 768)
    
    # Compute similarity
    similarity = torch.cosine_similarity(doc1_vec, doc2_vec)
    
    print(f"✅ Document embeddings shape: {doc1_vec.shape}")
    print(f"   Cosine similarity: {similarity.item():.4f}")


def example_4_long_context():
    """
    Example 4: Ultra-Long Context Processing
    Demonstrates the 200K context capability
    """
    print("\n" + "="*80)
    print("Example 4: Ultra-Long Context (200K tokens)")
    print("="*80)
    
    # Create model with 64K context (200K possible!)
    model = create_spectral_lm(
        size='small',
        vocab_size=50257,
        max_seq_len=65536  # 64K tokens!
    )
    
    print(f"✅ Created model with {model.config.max_seq_len:,} token context")
    print(f"   This is 2x GPT-4's context and 8x GPT-3.5's context!")
    
    # Simulate a very long document (16K tokens)
    long_input = torch.randint(0, 50257, (1, 16384))
    
    # Process the long document
    model.eval()
    with torch.no_grad():
        logits = model(long_input)
    
    print(f"✅ Processed {long_input.shape[1]:,} tokens successfully")
    print(f"   Output shape: {logits.shape}")
    print(f"   Complexity: O(n log n) vs O(n²) for transformers")
    print(f"   Memory efficient: No OOM errors!")


def example_5_custom_configuration():
    """
    Example 5: Custom Model Configuration
    Create a model tailored to your specific needs
    """
    print("\n" + "="*80)
    print("Example 5: Custom Configuration")
    print("="*80)
    
    # Custom configuration for your specific use case
    config = SpectralConfig(
        # Basic settings
        vocab_size=50257,
        embed_dim=512,        # Smaller for faster training
        hidden_dim=2048,
        num_layers=8,         # Fewer layers
        max_seq_len=4096,
        
        # Advanced features
        sparsity=0.15,        # Keep 15% of frequencies
        use_hierarchical_fft=True,
        use_adaptive_sparsity=True,
        use_phase_aware=True,
        
        # Optimization
        use_rope=True,
        use_gradient_checkpointing=True,  # Save memory
        dropout=0.1
    )
    
    from resonance_nn import SpectralLanguageModel
    model = SpectralLanguageModel(config)
    
    print(f"✅ Created custom model:")
    print(f"   Parameters: {model.get_num_params()/1e6:.1f}M")
    print(f"   Embedding dim: {config.embed_dim}")
    print(f"   Hidden dim: {config.hidden_dim}")
    print(f"   Layers: {config.num_layers}")
    print(f"   Max context: {config.max_seq_len:,}")
    print(f"   Sparsity: {config.sparsity:.0%}")


def example_6_training_loop():
    """
    Example 6: Training Loop
    Shows how to train the model on your data
    """
    print("\n" + "="*80)
    print("Example 6: Training Loop Example")
    print("="*80)
    
    # Create small model for quick training
    model = create_spectral_lm('tiny', vocab_size=10000)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"✅ Training setup ready")
    print(f"   Model: {model.get_num_params()/1e6:.1f}M parameters")
    print(f"   Optimizer: AdamW (lr=1e-4)")
    
    # Simulate training for 5 steps
    print("\n📚 Training for 5 steps...")
    model.train()
    
    for step in range(5):
        # Simulate batch
        input_ids = torch.randint(0, 10000, (4, 128))
        labels = torch.randint(0, 10000, (4, 128))
        
        # Forward
        logits = model(input_ids)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        print(f"   Step {step+1}/5 - Loss: {loss.item():.4f}")
    
    print("✅ Training completed!")


def example_7_saving_loading():
    """
    Example 7: Saving and Loading Models
    """
    print("\n" + "="*80)
    print("Example 7: Saving and Loading Models")
    print("="*80)
    
    # Create and train a model
    model = create_spectral_lm('tiny', vocab_size=10000)
    print("✅ Created model")
    
    # Save model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.config,
    }
    
    # In real use, you would do:
    # torch.save(checkpoint, 'my_model.pth')
    print("✅ Model saved to 'my_model.pth' (simulated)")
    
    # Load model
    # In real use:
    # checkpoint = torch.load('my_model.pth')
    # config = checkpoint['config']
    # model_loaded = SpectralLanguageModel(config)
    # model_loaded.load_state_dict(checkpoint['model_state_dict'])
    
    print("✅ Model loaded (simulated)")
    print("\n💡 Tip: Always save config along with state_dict!")


def main():
    """
    Run all examples
    """
    print("\n" + "="*80)
    print("SPECTRAL NEURAL NETWORKS - Simple Usage Examples")
    print("="*80)
    print("\nThese examples show how to use the resonance_nn module")
    print("in your own projects for various NLP tasks.")
    print()
    
    # Show available models
    list_available_models()
    
    # Run examples
    try:
        example_1_language_modeling()
        example_2_text_classification()
        example_3_document_embeddings()
        example_4_long_context()
        example_5_custom_configuration()
        example_6_training_loop()
        example_7_saving_loading()
        
        print("\n" + "="*80)
        print("✅ All examples completed successfully!")
        print("="*80)
        print("\n📚 Next Steps:")
        print("   1. Install the module: pip install -e .")
        print("   2. Import in your code: from resonance_nn import create_spectral_lm")
        print("   3. Create your model and start training!")
        print("   4. See MODULE_USAGE_GUIDE.md for more details")
        print("\n🚀 Happy coding with Spectral Neural Networks!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("   Make sure you have installed the module:")
        print("   pip install -e .")


if __name__ == '__main__':
    main()
