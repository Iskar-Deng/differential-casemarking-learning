# transformation.py 结构设计

## extract_svo_phrases
从句子中提取出主语 / 谓语(非be) / 宾语完整短语 (基本结构)
示例：
- He thinks that he is smart.
→ He / thinks / that he is smart

- I look for a job.
→ I / look for / a job

- She told me to leave.
→ She / told / me / to leave(补语)

- The dog is chasing the cat.
→ The dog / is chasing / the cat