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
