# 模块划分：Relational Animacy-Based Case Marking 转写系统

本系统将实现英语句子的结构转换、格标记与语序改写，支持方案 A / B / C。以下为主要模块划分。

---

## 1. `parser.py` – 结构分析器
- 使用 spaCy 提取：
  - 主语（nsubj / nsubj:pass）
  - 谓词（root 动词 + aux）
  - 宾语（obj / iobj）
  - 被动标记、从句类型、关系从句等依存关系

---

## 2. `animacy.py` – Animacy 判定器
- 词表 + WordNet 超义词规则
- 判断 noun 的生命度等级（Human > Animal > Movable Inanimate > ...）
- 标注主语、宾语 animacy 等级，为 case assignment 做准备

---

## 3. `transformation.py` – 结构转换器
- 实现三种转写策略（方案 A / B / C）：
  - A：仅加格标记
  - B：加格标记 + NMLZ
  - C：加格标记 + NMLZ + SOV改写 + 关系从句前置
- 子功能：
  - 被动句转主动
  - 名物化识别与替换（to V, V-ing, that-clause）
  - 定语从句识别与前置

---

## 4. `rewrite.py` – 改写输出器
- 根据结构变换结果，输出重写句
- 插入 AGE / PAT / NMLZ 标记
- 保留原始句，输出为结构化 `.tsv` 或 `.jsonl`

---

## 5. `main.py` – 批量处理调度器
- 加载输入句子文件
- 调用解析 + animacy + 转写 + 输出模块
- 保存结果用于训练 / 验证
