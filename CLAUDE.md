# Claude Code Project Instructions

## Code Dependency Analysis

コード依存関係は `.serena/memories/code_dependencies.md` に保存されています。

新しいセッションでは：
```
list_memories → read_memory("code_dependencies.md")
```

## Project Overview

食品画像に対する脳活動のfMRI分析プロジェクト。

### 主要分析パイプライン
1. **Behavior GLM**: 食品の主観的価値に関連する脳活動
2. **DNN GLM (Hierarchical)**: CLIP/ConvNeXtの階層的特徴量と脳活動の関連
3. **RSA**: 表象類似性分析によるDNN-脳対応の検証

### タスク実行
```bash
task --list  # 全タスク表示
```

## Serena Memory

プロジェクト情報は `.serena/memories/` に保存されています：
- `code_dependencies.md` - コード依存関係マップ
- `project_overview.md` - プロジェクト概要
