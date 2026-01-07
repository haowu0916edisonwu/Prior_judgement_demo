#!/usr/bin/env python3
"""
测试数据加载器 - 使用项目真实数据
"""

import sys
from pathlib import Path

# 导入项目的数据加载器
from src.data_loader import CAREDataLoader


def test_data_loading():
    """测试数据加载"""
    
    print("=" * 70)
    print("🧪 测试数据加载 - 使用项目数据")
    print("=" * 70)
    
    # 创建数据加载器（使用项目默认路径）
    loader = CAREDataLoader(data_root="data_care/eval", verbose=True)
    
    # 测试所有数据集
    datasets = ['nq', 'triviaqa', 'webqa', 'truthfulqa', 'factkg']
    
    print("\n" + "=" * 70)
    print("📊 测试所有数据集")
    print("=" * 70)
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"🔍 测试 {dataset.upper()}")
        print(f"{'='*70}")
        
        try:
            # 加载数据集（只加载前5个样本用于测试）
            samples = loader.load_dataset(dataset)
            
            if samples and len(samples) > 0:
                # 显示第一个样本的详细信息
                sample = samples[0]
                
                print(f"\n📦 第一个样本详情:")
                print(f"  ID: {sample.id}")
                print(f"  Question: {sample.question[:80]}...")
                print(f"  Answers: {sample.answers}")
                print(f"  Context length: {len(sample.top1_context)} chars")
                print(f"  Context preview: {sample.top1_context[:150]}...")
                
                results[dataset] = {
                    'status': '✅ 成功',
                    'samples': len(samples),
                    'first_sample': {
                        'id': sample.id,
                        'question': sample.question[:50],
                        'answers': sample.answers,
                        'context_length': len(sample.top1_context)
                    }
                }
                
                print(f"\n✅ 成功加载 {len(samples)} 个样本")
                
            else:
                results[dataset] = {'status': '❌ 失败', 'error': '未加载到样本'}
                print(f"❌ 未加载到样本")
                
        except Exception as e:
            results[dataset] = {'status': '❌ 失败', 'error': str(e)}
            print(f"❌ 加载失败: {e}")
    
    # 显示汇总
    print("\n" + "=" * 70)
    print("📊 测试汇总")
    print("=" * 70)
    
    success_count = 0
    for dataset, result in results.items():
        status = result['status']
        if '✅' in status:
            success_count += 1
            samples = result.get('samples', 0)
            print(f"{dataset:12s} {status:8s} - {samples:>5} samples")
        else:
            error = result.get('error', 'Unknown error')
            print(f"{dataset:12s} {status:8s} - {error}")
    
    print("\n" + "=" * 70)
    
    if success_count == len(datasets):
        print("✅ 所有测试通过！数据加载器工作正常。")
    else:
        print(f"⚠️  {success_count}/{len(datasets)} 个数据集测试通过")
    
    print("=" * 70)
    
    return success_count == len(datasets)


if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)