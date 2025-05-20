# 📘 修订版 DOM 规则（使用后置格标记）

## 📘 规则集一：简化的“仅基于生命度”的 DOM（3 个等级）

在这一规则集中，DOM（差别宾格标记）仅基于一个简化的生命度层级，分为 **三个等级**：

### 🔢 生命度等级（从高到低）：

1. 人类（Human）  
2. 动物（Animal）  
3. 无生命体（Inanimate）

### 🧾 DOM 规则：
- 如果**主语的生命度不高于宾语**，则：
  - 给主语添加 `AGE`（施事标记）
  - 给宾语添加 `PAT`（受事标记）
- 否则，不添加任何标记。

### ✅ 示例：

| 句子 | 主语生命度 | 宾语生命度 | 改写结果 |
|------|------------|------------|----------|
| I met **a teacher**. | 人类 | 人类 | I AGE met a teacher PAT. |
| The dog bit **the man**. | 动物 | 人类 | The dog AGE bit the man PAT. |
| The teacher praised **the dog**. | 人类 | 动物 | The teacher praised the dog. |
| The robot pushed **a car**. | 无生命体 | 无生命体 | The robot AGE pushed a car PAT. |
| A lion chased **a zebra**. | 动物 | 动物 | A lion AGE chased a zebra PAT. |

---

## 📗 规则集二：结合生命度与特指性的 DOM

此规则将**生命度（animacy）**和**特指性（definiteness）**结合，构建更细致的标记体系。

### 🔢 综合层级（从高到低）：

1. 第一/第二人称代词  
2. 第三人称代词  
3. 人类专有名词  
4. 明确的人类名词短语  
5. 特指的人类不定名词  
6. 明确的动物名词短语  
7. 明确的无生命名词  
8. 不定无生命名词

### 🧾 DOM 规则：
- 根据该综合等级，判断主语和宾语的**显著性**。
- 如果主语**不比宾语更显著**，则：
  - 给主语添加 `AGE`
  - 给宾语添加 `PAT`
- 否则，不添加任何标记。

### ✅ 示例：

| 句子 | 主语类型 | 宾语类型 | 改写结果 |
|------|----------|----------|----------|
| I met **Mary**. | 第一人称代词 | 专有名词 | I met Mary. |
| Mary met **me**. | 专有名词 | 第一人称代词 | Mary AGE met me PAT. |
| The teacher saw **a lion**. | 明确人类名词 | 明确动物名词 | The teacher saw a lion. |
| A dog chased **the teacher**. | 动物 | 明确人类名词 | A dog AGE chased the teacher PAT. |
| A machine moved **a box**. | 不定无生命 | 不定无生命 | A machine AGE moved a box PAT. |

---

## ✍️ 注释说明：

- `AGE` 标记施事，当它**不比宾语更显著或更高生命度**时；
- `PAT` 标记宾语，当它的显著性**不低于施事**时；
- 该逻辑基于**相对显著性（relational prominence）**，而非绝对等级；
- 本规则适用于类型学建模、构拟语言、或实验句法系统。
