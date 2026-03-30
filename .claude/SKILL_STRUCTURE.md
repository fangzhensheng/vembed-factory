# VEmbed Skills 结构说明

## ✅ vembed-quality Skill

规范的 Claude Code Skill 实现，用于项目 CI/CD 质量检查。

### 文件结构

```
.claude/skills/vembed-quality/
├── SKILL.md                      # 主 Skill 定义 (必需)
│   ├── YAML frontmatter          # name, description, compatibility
│   └── Markdown 使用指南         # 完整说明和示例
├── scripts/
│   └── vembed-ci-check.sh        # 可执行 CI 检查脚本
└── references/
    ├── tools.md                  # 工具参考文档
    └── config.md                 # 配置文档参考
```

### YAML Frontmatter 格式

```yaml
---
name: vembed-quality                    # Skill 唯一标识
description: |                          # 何时使用 (触发条件)
  Run comprehensive code quality...     # - 包含关键词
  Execute this whenever...              # - "推进" Claude 使用
compatibility:                          # 依赖/要求
  - venv: required (python 3.10+)
  - tools: black, isort, ruff, pytest
  - os: macOS, Linux
---
```

### 关键原则

1. **Name (必需)**
   - Skill 唯一标识
   - kebab-case 格式
   - 与目录名一致

2. **Description (最重要)**
   - 明确何时使用
   - 包含用户可能使用的关键词
   - 用"Should use this when"的语气
   - 不要太保守，要"推进" Claude 使用
   - 100-200 字最优

3. **Compatibility (可选)**
   - 列出依赖项
   - 列出要求的环境/工具
   - 列出支持的 OS

### Markdown 结构

```markdown
# Skill Title

Short description of what the skill does.

## When to Use This Skill

- Bullet points with use cases
- Be specific and clear

## Quick Start

The simplest way to get started.

## Usage Modes

Different ways to invoke the skill.

## What Each Tool Does

Explain each component.

## Common Workflows

Step-by-step examples.

## Troubleshooting

Common problems and solutions.

## See Also

Links to related resources.
```

### 渐进式披露原则

三个加载层级：

1. **Metadata** (~100 words)
   - name + description
   - 始终在上下文中
   - Claude 用这个决定是否使用 skill

2. **SKILL.md Body** (<500 lines)
   - 完整使用指南
   - Skill 被触发时加载
   - 包含示例和工作流

3. **References** (无限制)
   - 深度文档
   - 按需加载
   - 脚本可执行而无需加载

### Scripts 目录

- 包含可执行脚本 (`.sh`, `.py`)
- 脚本可以独立执行
- 无需加载进上下文即可运行
- 示例: vembed-ci-check.sh

### References 目录

- 详细文档和参考资料
- 当用户需要时加载
- 组织成逻辑分组
- 示例:
  - `tools.md` - 工具使用指南
  - `config.md` - 配置参考

## 与其他文档的关系

| 文件 | 用途 | 加载时机 |
|-----|------|---------|
| SKILL.md | Skill 主定义 | Skill 触发时 |
| scripts/*.sh | 自动化脚本 | 需要时手动执行 |
| references/*.md | 深度文档 | 用户查询时加载 |
| CI_QUALITY_REPORT.md | CI 验证报告 | 用户查询 |
| USAGE_GUIDE.md | 中文使用指南 | 参考用 |

## Description 优化建议

好的 description 特征：
- ✅ 包含具体关键词 (code review, CI/CD, commit)
- ✅ 明确调用语气 ("Whenever", "Make sure to")
- ✅ 列出主要功能
- ✅ 包含使用场景
- ❌ 不要含糊其辞 ("might be useful for")
- ❌ 不要太复杂 (保持简洁)

### 示例 Description

**好**:
```
Run comprehensive code quality checks for vembed-factory project.
Execute this whenever the user mentions code review, CI/CD validation,
testing, or preparing code for commit/merge. Use this skill to verify
code formatting (Black), import ordering (Isort), linting (Ruff), and
optionally run unit tests with coverage reports.
```

**不够好**:
```
Runs quality checks. Can check formatting and linting.
```

## 下一步改进

如果要优化 description 触发准确性，可以：

1. 生成 20 个 trigger eval queries (应该/不应该触发)
2. 用 Claude 测试当前 description
3. 运行优化循环 (`run_loop.py`)
4. 选择最佳 description

详见 skill-creator 文档中的 "Description Optimization" 部分。

## 参考资源

- **Skill Creator Guide**: `.claude/skills/skill-creator/`
- **Python Coding Standards**: `.claude/skills/python-coding-standards.md`
- **项目 CI/CD**: `.github/workflows/ci.yml`
