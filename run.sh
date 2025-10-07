#!/bin/bash

nohup python -m perturbation.run_perturb \
  --strategy local \
  --A_mode p3 --A_markedness forward \
  --P_mode none --P_markedness forward \
  > perturb_local_p3_forward.log 2>&1 &

nohup python -m perturbation.run_perturb \
  --strategy local \
  --A_mode animal --A_markedness forward \
  --P_mode none --P_markedness forward \
  > perturb_local_animal_forward.log 2>&1 &
