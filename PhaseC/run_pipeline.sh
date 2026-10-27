#!/usr/bin/env bash
# Phase C pipeline launcher. Stages are independent and resumable; every script skips finished work.
#
#   PhaseC/run_pipeline.sh status                      # which splits / checkpoints are available
#   PhaseC/run_pipeline.sh extract  [backbone ...]     # GPU: embeddings          (default: all five)
#   PhaseC/run_pipeline.sh adapters [backbone ...]     # CPU: 10 adapters each
#   PhaseC/run_pipeline.sh audit    [backbone ...]     # CPU: the audit (long; one job per backbone is the unit of parallelism)
#   PhaseC/run_pipeline.sh report                      # manifest + FINDINGS.md
#   PhaseC/run_pipeline.sh smoke    [backbone]         # quick audit on whatever exists, 200k pairs
#
# Environment: ID_SET (default cifar10), PY (python), SRUN_GPU / SRUN_CPU (set to "" to run in place),
# EXTRA (extra flags for the python script, e.g. EXTRA="--quick --n-pairs 200000").
# To submit the five audits as parallel jobs:  for b in $(...); do sbatch --wrap "PhaseC/run_pipeline.sh audit $b"; done
set -euo pipefail
cd "$(dirname "$0")/.."
source /data2/shared/apps/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-env 2>/dev/null || true

PY=${PY:-python3}
ID_SET=${ID_SET:-cifar10}
SRUN_GPU=${SRUN_GPU-srun --gres=gpu:1 -c 8 --mem=32G}
SRUN_CPU=${SRUN_CPU-srun -c 16 --mem=64G}
EXTRA=${EXTRA:-}
ALL="resnet18_cifar_openood resnet50_imagenet vitb16_imagenet dinov2_vitb14 clip_vitb16"
A=PhaseC/analysis

stage=${1:-status}; shift || true
BBS=${*:-$ALL}

case "$stage" in
  status)
    $PY $A/datasets_phasec.py --status
    $PY $A/backbones.py --list
    ;;
  extract)
    for b in $BBS; do $SRUN_GPU $PY $A/extract_embeddings.py --backbone "$b" --id-set "$ID_SET" --device cuda $EXTRA; done
    ;;
  adapters)
    for b in $BBS; do $SRUN_CPU $PY $A/train_adapters.py --backbone "$b" --id-set "$ID_SET"; done
    ;;
  audit)
    for b in $BBS; do $SRUN_CPU $PY $A/run_audit.py --backbone "$b" --id-set "$ID_SET" $EXTRA; done
    ;;
  smoke)
    b=${1:-resnet50_imagenet}
    $SRUN_CPU $PY $A/run_audit.py --backbone "$b" --id-set "$ID_SET" --quick --n-pairs 200000 --max-ood 3000 --max-train 10000 \
      --out PhaseC/outputs/smoke
    ;;
  report)
    $PY $A/write_manifest.py
    $PY $A/write_findings.py --id-set "$ID_SET"
    ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac
