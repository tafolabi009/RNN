"""
SuperGLUE Benchmark Suite for Spectral Neural Networks

Evaluates on 8 SuperGLUE tasks:
- BoolQ (Boolean Questions)
- CB (CommitmentBank)
- COPA (Choice of Plausible Alternatives)
- MultiRC (Multi-Sentence Reading Comprehension)
- ReCoRD (Reading Comprehension with Commonsense Reasoning Dataset)
- RTE (Recognizing Textual Entailment)
- WiC (Words in Context)
- WSC (Winograd Schema Challenge)
"""

import os
import time
import json
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resonance_nn import SpectralConfig, SpectralClassifier, SpectralSeq2Seq, ModalityType


@dataclass
class SuperGLUEResult:
    """Results for a single SuperGLUE task"""
    task: str
    metric: str
    score: float
    num_examples: int
    time_seconds: float
    params_millions: float
    additional_metrics: Optional[Dict] = None


class SuperGLUEBenchmark:
    """
    SuperGLUE Benchmark Suite
    
    More challenging than GLUE, tests:
    - Natural language understanding
    - Reasoning
    - Common sense
    - Coreference resolution
    
    Usage:
        benchmark = SuperGLUEBenchmark(model_size='large')
        results = benchmark.run_all_tasks()
        benchmark.save_results(results, 'superglue_results.json')
    """
    
    TASKS = {
        'boolq': {'metric': 'accuracy', 'num_labels': 2, 'type': 'classification'},
        'cb': {'metric': 'f1_accuracy', 'num_labels': 3, 'type': 'classification'},
        'copa': {'metric': 'accuracy', 'num_labels': 2, 'type': 'multiple_choice'},
        'multirc': {'metric': 'f1_em', 'num_labels': 2, 'type': 'classification'},
        'record': {'metric': 'f1_em', 'type': 'extractive_qa'},
        'rte': {'metric': 'accuracy', 'num_labels': 2, 'type': 'classification'},
        'wic': {'metric': 'accuracy', 'num_labels': 2, 'type': 'classification'},
        'wsc': {'metric': 'accuracy', 'num_labels': 2, 'type': 'classification'},
    }
    
    def __init__(
        self,
        model_size: str = 'large',
        batch_size: int = 16,
        max_seq_len: int = 512,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        pretrained_path: str = None,
    ):
        self.model_size = model_size
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.pretrained_path = pretrained_path
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Model cache
        self.models = {}
        
    def _create_config(self) -> SpectralConfig:
        """Create model config based on size"""
        if self.model_size == 'base':
            return SpectralConfig(
                vocab_size=self.tokenizer.vocab_size,
                num_layers=12,
                num_heads=12,
                hidden_dim=768,
                embed_dim=768,
                max_seq_len=self.max_seq_len,
                modality=ModalityType.TEXT,
            )
        elif self.model_size == 'large':
            return SpectralConfig(
                vocab_size=self.tokenizer.vocab_size,
                num_layers=24,
                num_heads=16,
                hidden_dim=1024,
                embed_dim=1024,
                max_seq_len=self.max_seq_len,
                modality=ModalityType.TEXT,
            )
        elif self.model_size == 'xlarge':
            return SpectralConfig(
                vocab_size=self.tokenizer.vocab_size,
                num_layers=32,
                num_heads=20,
                hidden_dim=1280,
                embed_dim=1280,
                max_seq_len=self.max_seq_len,
                modality=ModalityType.TEXT,
            )
        else:
            raise ValueError(f"Unknown model size: {self.model_size}")
    
    def _create_model(self, task: str) -> nn.Module:
        """Create model for specific task"""
        task_info = self.TASKS[task]
        config = self._create_config()
        
        if task_info['type'] == 'extractive_qa':
            # Use Seq2Seq for ReCoRD
            model = SpectralSeq2Seq(config)
        else:
            # Use Classifier for other tasks
            num_labels = task_info.get('num_labels', 2)
            model = SpectralClassifier(config, num_classes=num_labels)
        
        # Load pretrained weights if available
        if self.pretrained_path:
            checkpoint_path = os.path.join(self.pretrained_path, f'{task}_best.pt')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded pretrained weights for {task}")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _prepare_boolq_data(self, split: str = 'validation') -> DataLoader:
        """Prepare BoolQ dataset"""
        dataset = load_dataset('super_glue', 'boolq', split=split)
        
        def tokenize_fn(examples):
            texts = [f"{q} {p}" for q, p in zip(examples['question'], examples['passage'])]
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_seq_len,
            )
            encodings['labels'] = examples['label']
            return encodings
        
        tokenized = dataset.map(tokenize_fn, batched=True)
        tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return DataLoader(tokenized, batch_size=self.batch_size, shuffle=False)
    
    def _prepare_cb_data(self, split: str = 'validation') -> DataLoader:
        """Prepare CB (CommitmentBank) dataset"""
        dataset = load_dataset('super_glue', 'cb', split=split)
        
        def tokenize_fn(examples):
            encodings = self.tokenizer(
                examples['premise'],
                examples['hypothesis'],
                truncation=True,
                padding='max_length',
                max_length=self.max_seq_len,
            )
            encodings['labels'] = examples['label']
            return encodings
        
        tokenized = dataset.map(tokenize_fn, batched=True)
        tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return DataLoader(tokenized, batch_size=self.batch_size, shuffle=False)
    
    def _prepare_copa_data(self, split: str = 'validation') -> DataLoader:
        """Prepare COPA dataset"""
        dataset = load_dataset('super_glue', 'copa', split=split)
        
        def tokenize_fn(examples):
            # For each example, create two inputs (premise + choice1, premise + choice2)
            texts = []
            labels = []
            
            for premise, choice1, choice2, question, label in zip(
                examples['premise'], examples['choice1'], examples['choice2'],
                examples['question'], examples['label']
            ):
                connector = "because" if question == "cause" else "so"
                texts.append(f"{premise} {connector} {choice1}")
                texts.append(f"{premise} {connector} {choice2}")
                labels.append(label)
            
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_seq_len,
            )
            
            # Reshape to pairs
            for key in encodings:
                encodings[key] = [encodings[key][i:i+2] for i in range(0, len(encodings[key]), 2)]
            
            encodings['labels'] = labels
            return encodings
        
        tokenized = dataset.map(tokenize_fn, batched=True)
        
        return tokenized  # Return dataset directly, handle in evaluation
    
    def _prepare_wic_data(self, split: str = 'validation') -> DataLoader:
        """Prepare WiC (Words in Context) dataset"""
        dataset = load_dataset('super_glue', 'wic', split=split)
        
        def tokenize_fn(examples):
            texts = [
                f"{s1} {s2} [WORD] {w}"
                for s1, s2, w in zip(examples['sentence1'], examples['sentence2'], examples['word'])
            ]
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_seq_len,
            )
            encodings['labels'] = examples['label']
            return encodings
        
        tokenized = dataset.map(tokenize_fn, batched=True)
        tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return DataLoader(tokenized, batch_size=self.batch_size, shuffle=False)
    
    def _prepare_rte_data(self, split: str = 'validation') -> DataLoader:
        """Prepare RTE dataset"""
        dataset = load_dataset('super_glue', 'rte', split=split)
        
        def tokenize_fn(examples):
            encodings = self.tokenizer(
                examples['premise'],
                examples['hypothesis'],
                truncation=True,
                padding='max_length',
                max_length=self.max_seq_len,
            )
            encodings['labels'] = examples['label']
            return encodings
        
        tokenized = dataset.map(tokenize_fn, batched=True)
        tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return DataLoader(tokenized, batch_size=self.batch_size, shuffle=False)
    
    def _prepare_wsc_data(self, split: str = 'validation') -> DataLoader:
        """Prepare WSC (Winograd Schema Challenge) dataset"""
        dataset = load_dataset('super_glue', 'wsc', split=split)
        
        def tokenize_fn(examples):
            # Highlight the span in the text
            texts = []
            for text, span1_idx, span2_idx, span1_text, span2_text in zip(
                examples['text'], examples['span1_index'], examples['span2_index'],
                examples['span1_text'], examples['span2_text']
            ):
                # Simple highlighting: add markers
                texts.append(f"{text} [SPAN1] {span1_text} [SPAN2] {span2_text}")
            
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_seq_len,
            )
            encodings['labels'] = examples['label']
            return encodings
        
        tokenized = dataset.map(tokenize_fn, batched=True)
        tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return DataLoader(tokenized, batch_size=self.batch_size, shuffle=False)
    
    def _compute_metrics(self, task: str, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        """Compute task-specific metrics"""
        metrics = {}
        
        if task in ['boolq', 'copa', 'wic', 'rte', 'wsc']:
            metrics['accuracy'] = accuracy_score(labels, predictions)
            metrics['score'] = metrics['accuracy']
        
        elif task == 'cb':
            metrics['accuracy'] = accuracy_score(labels, predictions)
            metrics['f1'] = f1_score(labels, predictions, average='macro')
            metrics['score'] = (metrics['accuracy'] + metrics['f1']) / 2
        
        elif task in ['multirc', 'record']:
            # F1 and Exact Match
            metrics['f1'] = f1_score(labels, predictions, average='binary')
            metrics['em'] = accuracy_score(labels, predictions)
            metrics['score'] = (metrics['f1'] + metrics['em']) / 2
        
        return metrics
    
    def evaluate_task(self, task: str) -> SuperGLUEResult:
        """Evaluate on a single SuperGLUE task"""
        print(f"\n{'='*80}")
        print(f"Evaluating {task.upper()}")
        print(f"{'='*80}")
        
        # Create model
        if task not in self.models:
            self.models[task] = self._create_model(task)
        model = self.models[task]
        
        params_millions = sum(p.numel() for p in model.parameters()) / 1e6
        
        # Load data (task-specific)
        if task == 'boolq':
            dataloader = self._prepare_boolq_data()
        elif task == 'cb':
            dataloader = self._prepare_cb_data()
        elif task == 'copa':
            # COPA needs special handling
            dataset = self._prepare_copa_data()
            return self._evaluate_copa(model, dataset, params_millions)
        elif task == 'wic':
            dataloader = self._prepare_wic_data()
        elif task == 'rte':
            dataloader = self._prepare_rte_data()
        elif task == 'wsc':
            dataloader = self._prepare_wsc_data()
        else:
            print(f"⚠ Task {task} not yet implemented, skipping...")
            return None
        
        # Evaluate
        all_predictions = []
        all_labels = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task}"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = model(input_ids)
                predictions = logits.argmax(dim=-1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        elapsed_time = time.time() - start_time
        
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        
        # Compute metrics
        metrics = self._compute_metrics(task, all_predictions, all_labels)
        
        result = SuperGLUEResult(
            task=task,
            metric=self.TASKS[task]['metric'],
            score=metrics['score'],
            num_examples=len(all_labels),
            time_seconds=elapsed_time,
            params_millions=params_millions,
            additional_metrics=metrics,
        )
        
        print(f"\n✓ {task.upper()} Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"  Examples: {result.num_examples:,}")
        print(f"  Time: {result.time_seconds:.2f}s")
        
        return result
    
    def _evaluate_copa(self, model, dataset, params_millions):
        """Special evaluation for COPA (multiple choice)"""
        all_predictions = []
        all_labels = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for example in tqdm(dataset, desc="Evaluating COPA"):
                # Get scores for both choices
                input_ids_1 = torch.tensor(example['input_ids'][0]).unsqueeze(0).to(self.device)
                input_ids_2 = torch.tensor(example['input_ids'][1]).unsqueeze(0).to(self.device)
                
                logits_1 = model(input_ids_1)
                logits_2 = model(input_ids_2)
                
                # Choose the one with higher positive class score
                score_1 = logits_1[0, 1].item()
                score_2 = logits_2[0, 1].item()
                
                prediction = 0 if score_1 > score_2 else 1
                all_predictions.append(prediction)
                all_labels.append(example['labels'])
        
        elapsed_time = time.time() - start_time
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        metrics = self._compute_metrics('copa', all_predictions, all_labels)
        
        return SuperGLUEResult(
            task='copa',
            metric='accuracy',
            score=metrics['score'],
            num_examples=len(all_labels),
            time_seconds=elapsed_time,
            params_millions=params_millions,
            additional_metrics=metrics,
        )
    
    def run_all_tasks(self) -> Dict[str, SuperGLUEResult]:
        """Run all SuperGLUE tasks"""
        print(f"\n{'='*80}")
        print(f"SuperGLUE BENCHMARK - Spectral Neural Networks v3.0")
        print(f"{'='*80}")
        print(f"Model size: {self.model_size}")
        print(f"Device: {self.device}")
        
        results = {}
        
        for task in self.TASKS.keys():
            try:
                result = self.evaluate_task(task)
                if result:
                    results[task] = result
            except Exception as e:
                print(f"\n❌ Error evaluating {task}: {e}")
                import traceback
                traceback.print_exc()
        
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, SuperGLUEResult]):
        """Print benchmark summary"""
        print(f"\n{'='*80}")
        print(f"SuperGLUE BENCHMARK SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"{'Task':<15} {'Score':<10} {'Examples':<10} {'Time (s)':<10}")
        print("-" * 80)
        
        scores = []
        for task, result in results.items():
            print(f"{task.upper():<15} {result.score:<10.4f} {result.num_examples:<10,} {result.time_seconds:<10.2f}")
            scores.append(result.score)
        
        print("-" * 80)
        print(f"{'AVERAGE':<15} {np.mean(scores):<10.4f}")
        print(f"\n{'='*80}")
    
    def save_results(self, results: Dict[str, SuperGLUEResult], output_path: str):
        """Save results to JSON"""
        output_dict = {
            'model_size': self.model_size,
            'results': {task: asdict(result) for task, result in results.items()},
            'average_score': np.mean([r.score for r in results.values()]),
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_dict, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='SuperGLUE Benchmark')
    parser.add_argument('--model_size', type=str, default='large', choices=['base', 'large', 'xlarge'])
    parser.add_argument('--tasks', nargs='+', default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--output', type=str, default='superglue_results.json')
    
    args = parser.parse_args()
    
    benchmark = SuperGLUEBenchmark(
        model_size=args.model_size,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        pretrained_path=args.pretrained_path,
    )
    
    if args.tasks:
        results = {}
        for task in args.tasks:
            result = benchmark.evaluate_task(task)
            if result:
                results[task] = result
        benchmark._print_summary(results)
    else:
        results = benchmark.run_all_tasks()
    
    benchmark.save_results(results, args.output)


if __name__ == '__main__':
    main()
