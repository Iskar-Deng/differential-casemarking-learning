# 📘 项目计划：Relational Animacy-Based Case Marking 数据构造阶段  
**时间范围：2025年5月19日 – 5月25日**

---

## ✅ 阶段一：现象模板定型

### 🎯 目标  
制定并不断完善 case 标记规则模板，为实际语料处理提供统一标准。

### 🔧 任务  
- 定义 animacy 层级（Human > Animal > Movable Inanimate > Inanimate > Event）
- 明确 case 标记触发条件（S.animacy ≤ O.animacy）
- 设计 rewrite 格式（如：`cat=SUB`, `boy=OBJ`）
- 规范从句、结构异常的处理方式（如忽略宾语从句）
- 输出：规则文档、示例、问题记录表

---

## ✍️ 阶段二：数据构造与标注

### 🎯 目标  
构建高质量语料，标注结构信息并根据规则转写。

### 🔧 任务  
- 抽取 BabyLM 句子（目标 500–1,000 条）
- 使用 Stanza 分析主语/宾语/动词
- 按规则添加 case 标记
- 标注是否正确并记录例外情况
- 输出：原句、改写句、结构标注数据表

---

## 🧪 阶段三：数据拆分与验证

### 🎯 目标  
整理成训练/验证/测试集，构建评估输入。

### 🔧 任务  
- 拆分数据（70/15/15）
- 确保 animacy pair 分布均衡，测试集含 unseen 对比
- 构造 few-shot 推理输入（prompt 格式）
- 输出：train/dev/test 数据集，evaluation 输入集

---
