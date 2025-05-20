# Relational Animacy-Based Case Marking 标注规范草案

## 1. Animacy 层级与判断机制

### 1.1 Animacy 层级定义
参考 Silverstein 层级并融合玛丽玛萨话语法数据，使用以下顺序：
- Human > Animal > Movable Inanimate > Inanimate > Abstract/Event

### 1.2 Definiteness 补充机制（可选）
在后期可引入 definiteness（特指性）作为次要判定因子，用于解决 animacy 等级相等时的歧义。

---

## 2. 三套标注逻辑
- A: 只改基本的SVO结构，系动词不加标记 
- B: 除了SVO结构，被动句，关系从句，名物化，形式主语，等都改成NMLZ标记的形式 
- C: 不仅加上格标记和NMZ标记，还把语序改成SOV的形式，同时定语前置

---

## 3. 从句与结构处理逻辑

### 3.1 名物化结构
以下结构均应去除英语功能词（如 that, to, -ing），使用 `NMLZ` 标记名物化边界：
- 关系从句：前置，内部加格标记，语序按方案处理
- 不定式（to V） → V NMLZ
- 动名词（V-ing） → V NMLZ
- that/what等 从句 → 去掉 关系词，直接名物化子句块

### 3.2 形式主语
保留 A 方案原句；
B/C 方案中将结构名物化，例如：
- 原句：It is important to be kind.
- 方案 C：be kind =NMLZ important is.

### 3.3 被动句
被动结构统一转为主动表达（B/C）：
- 原句：The man was bitten (by the dog).
- 改写：The dog AGE the man PAT bit.
如果原句没有 agent ，改写后也不需要.

### 3.4 系动词
- 所有系动词（is, are, was, etc.）句型默认不加 case 标记；

### 3.5 省略结构
- 保留原来的结构，根据省略的论元来推测未省略的论元是否加 case 标记, 如(Give me the book).

---

## 3. 三套结构方案（A / B / C）

### 3.1 区别说明

- A：仅标注基本 SVO 中的 AGE / PAT，不添加 NMLZ，不调语序；
- B：添加 NMLZ 标记，处理从句但不调语序；
- C：添加 NMLZ、格标记，同时转换为 SOV，定语从句整体前置。

### 3.2 标记规范

- `AGE`、`PAT`：独立 token，紧跟在名词后；
- `NMLZ`：附加在整个子句末尾，表示结构已名物化；
- 所有转写句中，保留完整空格间隔格式；
- 被动句恢复为主动，形式主语结构用 NMLZ 替换。

---

## 4. 示例集

**原句**：The cat who bit the dog saw the boy.

- 方案 A：
  → The cat who AGE bit the dog PAT AGE saw the boy PAT.

- 方案 B：
  → The cat bit the dog PAT NMLZ AGE saw the boy PAT.

- 方案 C：
  → bit the dog PAT NMLZ the cat AGE the boy PAT saw.

---

## 5. 杂项与边界操作规范

### 5.1 形式还原
- 宾格主格还原：me → I, him → he, her → she
- 关系代词还原：whom → who

### 5.2 动词还原
- 所有 -ing 形式 → 原形：running → run
- 所有 to V 结构 → V NMLZ

## 5.3 主谓一致
- 是否需要还原？

---
