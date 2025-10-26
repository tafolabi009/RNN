"""
GLUE Benchmark Suite for Spectral Neural Networks

Evaluates on all 9 GLUE tasks:
- CoLA (Corpus of Linguistic Acceptability)
- SST-2 (Stanford Sentiment Treebank)
- MRPC (Microsoft Research Paraphrase Corpus)
- STS-B (Semantic Textual Similarity Benchmark)
- QQP (Quora Question Pairs)
- MNLI (Multi-Genre Natural Language Inference)
- QNLI (Question Natural Language Inference)
- RTE (Recognizing Textual Entailment)
- WNLI (Winograd Natural Language Inference)
"""

import os
import time
import json
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resonance_nn import SpectralConfig, SpectralClassifier, ModalityType


@dataclass
class BenchmarkResult:
    """Results for a single GLUE task"""
    task: str
    metric: str
    score: float
    num_examples: int
    time_seconds: float
    params_millions: float


class GLUEBenchmark:
    """
    GLUE Benchmark Suite
    
    Usage:
        benchmark = GLUEBenchmark(model_size='base')
        results = benchmark.run_all_tasks()
        benchmark.save_results(results, 'glue_results.json')
    """
    
    TASKS = {
        'cola': {'metric': 'matthews_corrcoef', 'num_labels': 2},
        'sst2': {'metric': 'accuracy', 'num_labels': 2},
        'mrpc': {'metric': 'f1', 'num_labels': 2},
        'stsb': {'metric': 'pearson_spearman', 'num_labels': 1, 'regression': True},
        'qqp': {'metric': 'f1', 'num_labels': 2},
        'mnli': {'metric': 'accuracy', 'num_labels': 3},
        'qnli': {'metric': 'accuracy', 'num_labels': 2},
        'rte': {'metric': 'accuracy', 'num_labels': 2},
        'wnli': {'metric': 'accuracy', 'num_labels': 2},
    }
    
    def __init__(
        self,
        model_size: str = 'base',
        batch_size: int = 32,
        max_seq_len: int = 512,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        pretrained_path: str = None,
    ):
        self.model_size = model_size
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.pretrained_path = pretrained_path
        
        # Initialize tokenizer (using BERT tokenizer for compatibility)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Model cache
        self.models = {}
        
    def _create_model(self, task: str) -> nn.Module:
        """Create model for specific task"""
        task_info = self.TASKS[task]
        num_labels = task_info['num_labels']
        
        # Create config
        if self.model_size == 'tiny':
            config = SpectralConfig(
                vocab_size=self.tokenizer.vocab_size,
                num_layers=6,
                num_heads=4,
                hidden_dim=256,
                embed_dim=256,
                max_seq_len=self.max_seq_len,
                modality=ModalityType.TEXT,
            )
        elif self.model_size == 'base':
            config = SpectralConfig(
                vocab_size=self.tokenizer.vocab_size,
                num_layers=12,
                num_heads=12,
                hidden_dim=768,
                embed_dim=768,
                max_seq_len=self.max_seq_len,
                modality=ModalityType.TEXT,
            )
        elif self.model_size == 'large':
            config = SpectralConfig(
                vocab_size=self.tokenizer.vocab_size,
                num_layers=24,
                num_heads=16,
                hidden_dim=1024,
                embed_dim=1024,
                max_seq_len=self.max_seq_len,
                modality=ModalityType.TEXT,
            )
        else:
            raise ValueError(f"Unknown model size: {self.model_size}")
        
        # Create classifier
        model = SpectralClassifier(config, num_classes=num_labels)
        
        # Load pretrained weights if available
        if self.pretrained_path:
            checkpoint = torch.load(
                os.path.join(self.pretrained_path, f'{task}_best.pt'),
                map_location=self.device
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded pretrained weights for {task}")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _prepare_data(self, task: str, split: str = 'validation') -> DataLoader:
        """Load and prepare GLUE dataset"""
        
        # Map task names to dataset names
        dataset_map = {
            'mnli': 'glue',
            'sst2': 'glue',
            'cola': 'glue',
            'mrpc': 'glue',
            'stsb': 'glue',
            'qqp': 'glue',
            'qnli': 'glue',
            'rte': 'glue',
            'wnli': 'glue',
        }
        
        # Load dataset
        if split == 'validation' and task == 'mnli':
            # MNLI has matched and mismatched validation sets
            dataset = load_dataset(dataset_map[task], task, split='validation_matched')
        else:
            dataset = load_dataset(dataset_map[task], task, split=split)
        
        # Tokenize
        def tokenize_function(examples):
            # Handle different input formats
            if task in ['mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']:
                # Sentence pair tasks
                texts = list(zip(examples['sentence1'], examples['sentence2']))
                encodings = self.tokenizer(
                    [t[0] for t in texts],
                    [t[1] for t in texts],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_seq_len,
                )
            else:
                # Single sentence tasks
                encodings = self.tokenizer(
                    examples['sentence'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_seq_len,
                )
            
            encodings['labels'] = examples['label']
            return encodings
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        
        return dataloader
    
    def _compute_metrics(self, task: str, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute task-specific metrics"""
        task_info = self.TASKS[task]
        metric_name = task_info['metric']
        
        if metric_name == 'accuracy':
            return (predictions == labels).mean()
        
        elif metric_name == 'f1':
            return f1_score(labels, predictions, average='binary')
        
        elif metric_name == 'matthews_corrcoef':
            return matthews_corrcoef(labels, predictions)
        
        elif metric_name == 'pearson_spearman':
            # For regression tasks (STS-B)
            pearson = pearsonr(predictions, labels)[0]
            spearman = spearmanr(predictions, labels)[0]
            return (pearson + spearman) / 2
        
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    def evaluate_task(self, task: str) -> BenchmarkResult:
        """Evaluate on a single GLUE task"""
        print(f"\n{'='*80}")
        print(f"Evaluating {task.upper()}")
        print(f"{'='*80}")
        
        # Create model
        if task not in self.models:
            self.models[task] = self._create_model(task)
        model = self.models[task]
        
        # Get params
        params_millions = sum(p.numel() for p in model.parameters()) / 1e6
        
        # Load data
        dataloader = self._prepare_data(task, split='validation')
        
        # Evaluate
        all_predictions = []
        all_labels = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task}"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = model(input_ids)
                
                # Get predictions
                if self.TASKS[task].get('regression', False):
                    predictions = logits.squeeze(-1)
                else:
                    predictions = logits.argmax(dim=-1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        elapsed_time = time.time() - start_time
        
        # Concatenate results
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        
        # Compute metric
        score = self._compute_metrics(task, all_predictions, all_labels)
        
        # Create result
        result = BenchmarkResult(
            task=task,
            metric=self.TASKS[task]['metric'],
            score=score,
            num_examples=len(all_labels),
            time_seconds=elapsed_time,
            params_millions=params_millions,
        )
        
        print(f"\n✓ {task.upper()} Results:")
        print(f"  Metric: {result.metric}")
        print(f"  Score: {result.score:.4f}")
        print(f"  Examples: {result.num_examples:,}")
        print(f"  Time: {result.time_seconds:.2f}s")
        print(f"  Throughput: {result.num_examples/result.time_seconds:.1f} ex/s")
        
        return result
    
    def run_all_tasks(self) -> Dict[str, BenchmarkResult]:
        """Run all GLUE tasks"""
        print(f"\n{'='*80}")
        print(f"GLUE BENCHMARK - Spectral Neural Networks v3.0")
        print(f"{'='*80}")
        print(f"Model size: {self.model_size}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max sequence length: {self.max_seq_len}")
        
        results = {}
        
        for task in self.TASKS.keys():
            try:
                result = self.evaluate_task(task)
                results[task] = result
            except Exception as e:
                print(f"\n❌ Error evaluating {task}: {e}")
                import traceback
                traceback.print_exc()
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, BenchmarkResult]):
        """Print benchmark summary"""
        print(f"\n{'='*80}")
        print(f"GLUE BENCHMARK SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"{'Task':<10} {'Metric':<20} {'Score':<10} {'Examples':<10} {'Time (s)':<10}")
        print("-" * 80)
        
        total_time = 0
        scores = []
        
        for task, result in results.items():
            print(f"{task.upper():<10} {result.metric:<20} {result.score:<10.4f} "
                  f"{result.num_examples:<10,} {result.time_seconds:<10.2f}")
            total_time += result.time_seconds
            scores.append(result.score)
        
        print("-" * 80)
        print(f"{'AVERAGE':<10} {'':<20} {np.mean(scores):<10.4f} {'':<10} {total_time:<10.2f}")
        print(f"\n{'='*80}")
    
    def save_results(self, results: Dict[str, BenchmarkResult], output_path: str):
        """Save results to JSON"""
        output_dict = {
            'model_size': self.model_size,
            'device': self.device,
            'batch_size': self.batch_size,
            'max_seq_len': self.max_seq_len,
            'results': {task: asdict(result) for task, result in results.items()},
            'average_score': np.mean([r.score for r in results.values()]),
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_dict, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='GLUE Benchmark for Spectral Neural Networks')
    parser.add_argument('--model_size', type=str, default='base', choices=['tiny', 'base', 'large'],
                       help='Model size')
    parser.add_argument('--tasks', nargs='+', default=None,
                       help='Specific tasks to run (default: all)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--max_seq_len', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained checkpoints')
    parser.add_argument('--output', type=str, default='glue_results.json',
                       help='Output path for results')
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = GLUEBenchmark(
        model_size=args.model_size,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        pretrained_path=args.pretrained_path,
    )
    
    # Run tasks
    if args.tasks:
        results = {}
        for task in args.tasks:
            results[task] = benchmark.evaluate_task(task)
        benchmark._print_summary(results)
    else:
        results = benchmark.run_all_tasks()
    
    # Save results
    benchmark.save_results(results, args.output)


if __name__ == '__main__':
    main()
