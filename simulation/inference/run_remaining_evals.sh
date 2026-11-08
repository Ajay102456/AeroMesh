#!/bin/bash
set -e
cd /home/s/Documents/AeroMesh/simulation/inference
source ../../venv/bin/activate

# wait for the currently-running nonpalu-on-test eval to finish naturally
while pgrep -f "evaluate.py --checkpoint model_nonpalu.pt --data_dir crops/test" > /dev/null; do sleep 5; done

python3 evaluate.py --checkpoint model_nonpalu.pt --data_dir crops/val --out eval_nonpalu_on_val.npz > eval_nonpalu_val.log 2>&1
python3 evaluate.py --checkpoint model_nonpalu.pt --data_dir crops/palu_test --out eval_nonpalu_on_palutest.npz > eval_nonpalu_palutest.log 2>&1
python3 evaluate.py --checkpoint model_nonpalu.pt --data_dir crops/palu_calib --out eval_nonpalu_on_palucalib.npz > eval_nonpalu_palucalib.log 2>&1
python3 evaluate.py --checkpoint model_palu.pt --data_dir crops/test --out eval_palu_on_test.npz > eval_palu_test.log 2>&1
python3 evaluate.py --checkpoint model_palu.pt --data_dir crops/val --out eval_palu_on_val.npz > eval_palu_val.log 2>&1
python3 evaluate.py --checkpoint model_palu.pt --data_dir crops/palu_test --out eval_palu_on_palutest.npz > eval_palu_palutest.log 2>&1
python3 evaluate.py --checkpoint model_palu.pt --data_dir crops/palu_calib --out eval_palu_on_palucalib.npz > eval_palu_palucalib.log 2>&1

echo "ALL_EVALS_DONE"
