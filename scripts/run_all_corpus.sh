############################
# Ⅰ. 单向系统（independent）
############################

# 3. A_none + P_definite
python -m perturbation.run_perturb_v2 \
  --system independent \
  --A_mode none \
  --P_mode definite


### —— 逆向版本（P不是该类才加）——

# 4. A_none + P_animate逆向
python -m perturbation.run_perturb_v2 \
  --system independent \
  --A_mode none \
  --P_mode animate \
  --inverse

# 5. A_none + P_pronoun逆向
python -m perturbation.run_perturb_v2 \
  --system independent \
  --A_mode none \
  --P_mode pronoun \
  --inverse

# 6. A_none + P_definite逆向
python -m perturbation.run_perturb_v2 \
  --system independent \
  --A_mode none \
  --P_mode definite \
  --inverse



############################
# Ⅱ. A-only 系列（independent）
############################

# 7. A_animate + P_none
python -m perturbation.run_perturb_v2 \
  --system independent \
  --A_mode animate \
  --P_mode none

# 8. A_pronoun + P_none
python -m perturbation.run_perturb_v2 \
  --system independent \
  --A_mode pronoun \
  --P_mode none

# 9. A_definite + P_none
python -m perturbation.run_perturb_v2 \
  --system independent \
  --A_mode definite \
  --P_mode none


### —— 逆向版本（A是该类才加）——

# 10. A_animate逆向 + P_none
python -m perturbation.run_perturb_v2 \
  --system independent \
  --A_mode animate \
  --P_mode none \
  --inverse

# 11. A_pronoun逆向 + P_none
python -m perturbation.run_perturb_v2 \
  --system independent \
  --A_mode pronoun \
  --P_mode none \
  --inverse

# 12. A_definite逆向 + P_none
python -m perturbation.run_perturb_v2 \
  --system independent \
  --A_mode definite \
  --P_mode none \
  --inverse



############################
# Ⅲ. 双向系统（parallel）
############################

# 13. A_animate + P_animate（A/P各自独立判断）
python -m perturbation.run_perturb_v2 \
  --system independent \
  --A_mode animate \
  --P_mode animate

# 14. A_pronoun + P_pronoun
python -m perturbation.run_perturb_v2 \
  --system independent \
  --A_mode pronoun \
  --P_mode pronoun

# 15. A_definite + P_definite
python -m perturbation.run_perturb_v2 \
  --system independent \
  --A_mode definite \
  --P_mode definite



############################
# Ⅳ. 双控系统（dualP）
############################

# 16. P同时是[+animate, +definite] 才加
python -m perturbation.run_perturb_v2 \
  --system dualP \
  --P_combo and

# 17. 逆向：P同时是[+animate, +definite] 才不加
python -m perturbation.run_perturb_v2 \
  --system dualP \
  --P_combo and \
  --inverse

# 18. P只要是[+animate, +definite] 其中一个就加
python -m perturbation.run_perturb_v2 \
  --system dualP \
  --P_combo or

# 19. 逆向：P只要是[+animate, +definite] 其中一个就不加
python -m perturbation.run_perturb_v2 \
  --system dualP \
  --P_combo or \
  --inverse



############################
# Ⅴ. 全局系统（global）
############################

# 20. A的animacy比P小就加
python -m perturbation.run_perturb_v2 \
  --system global \
  --compare_attr animacy \
  --direction up

# 21. A的animacy比P大就加
python -m perturbation.run_perturb_v2 \
  --system global \
  --compare_attr animacy \
  --direction down

# 22. A的nptype比P小就加
python -m perturbation.run_perturb_v2 \
  --system global \
  --compare_attr nptype \
  --direction up

# 23. A的nptype比P大就加
python -m perturbation.run_perturb_v2 \
  --system global \
  --compare_attr nptype \
  --direction down

# 24. A的definiteness比P小就加
python -m perturbation.run_perturb_v2 \
  --system global \
  --compare_attr definiteness \
  --direction up

# 25. A的definiteness比P大就加
python -m perturbation.run_perturb_v2 \
  --system global \
  --compare_attr definiteness \
  --direction down
