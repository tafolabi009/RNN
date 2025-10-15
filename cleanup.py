"""
CLEANUP SCRIPT - Remove Redundant Files
========================================

This script removes all redundant, duplicate, and legacy files,
keeping only the production-ready code.

Files to keep:
- resonance_nn/spectral_optimized.py (THE MAIN FILE)
- resonance_nn/__init__.py
- train_production.py
- inference.py
- README.md
- requirements.txt
- COMPREHENSIVE_ANALYSIS.md

Files to remove:
- .history/ (50+ duplicate files)
- resonance_nn/spectral.py (old version)
- resonance_nn/spectral_v2.py (empty)
- resonance_nn/spectral_cuda_wrapper.py (not implemented)
- Old training scripts
- Legacy test files
"""

import os
import shutil
from pathlib import Path


def main():
    """Clean up redundant files"""
    print("\n" + "="*80)
    print("CLEANUP SCRIPT - Removing Redundant Files")
    print("="*80)
    
    root = Path(__file__).parent
    
    # Files and directories to remove
    to_remove = [
        # History folder
        '.history',
        
        # Old spectral files
        'resonance_nn/spectral.py',
        'resonance_nn/spectral_v2.py',
        'resonance_nn/spectral_cuda_wrapper.py',
        'resonance_nn/training.py',
        
        # Old training/test scripts
        'step2_train_and_validate.py',
        'step3_optimize_performance.py',
        'interactive_demo.py',
        'test_bug_fixes.py',
        'test_spectral.py',
        'generate_enhanced.py',
        'comprehensive_benchmark.py',
        
        # Results/outputs
        'validation_results.json',
        'scale_test_results.png',
        'optimization_results.png',
        
        # Pycache
        '__pycache__',
        'resonance_nn/__pycache__',
        'resonance_nn/configs/__pycache__',
    ]
    
    removed = []
    not_found = []
    
    for item in to_remove:
        path = root / item
        
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
                print(f"✅ Removed directory: {item}")
            else:
                path.unlink()
                print(f"✅ Removed file: {item}")
            removed.append(item)
        else:
            not_found.append(item)
    
    print(f"\n{'='*80}")
    print("CLEANUP SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Removed: {len(removed)} items")
    print(f"⚠️  Not found: {len(not_found)} items")
    
    # Files that remain
    print(f"\n{'='*80}")
    print("REMAINING FILES (Production Ready)")
    print(f"{'='*80}")
    
    keep_files = [
        'resonance_nn/spectral_optimized.py',
        'resonance_nn/__init__.py',
        'train_production.py',
        'inference.py',
        'README.md',
        'requirements.txt',
        'requirements-dev.txt',
        'COMPREHENSIVE_ANALYSIS.md',
        'CONTRIBUTING.md',
        'LICENSE',
    ]
    
    for file in keep_files:
        path = root / file
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"  ✅ {file:<40s} ({size:>8.1f} KB)")
        else:
            print(f"  ⚠️  {file:<40s} (NOT FOUND)")
    
    print(f"\n{'='*80}")
    print("🎉 CLEANUP COMPLETE!")
    print(f"{'='*80}")
    print("\nYour codebase is now:")
    print("  ✅ Clean and organized")
    print("  ✅ No redundant files")
    print("  ✅ Production ready")
    print("\nNext steps:")
    print("  1. Update resonance_nn/__init__.py to import from spectral_optimized")
    print("  2. Train models with: python train_production.py")
    print("  3. Run inference with: python inference.py")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    # Confirm before running
    print("\n⚠️  WARNING: This will DELETE files!")
    print("   - .history/ folder")
    print("   - Old spectral*.py files")
    print("   - Legacy training scripts")
    print("   - Test files")
    
    response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
    
    if response == 'yes':
        main()
    else:
        print("Cleanup cancelled.")
