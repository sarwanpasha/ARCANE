#!/bin/bash
# Run all competitor baselines for one dataset/seed[/holdout], then evaluate.
# Usage (from ~/arcane): bash run_comp.sh <ds> <seed> [holdout]
# Requires emb/<tag>_seed<s>_{full,cf0,cf1}.npz from train.py for scanviEM, scHPLknn,
# ConformalSC-scvi and the (scVI, scPoli) ARCANE pair. Each step is independent: a failure is
# recorded in res/comp/<method>_<tag>_seed<s>.error.json and the loop continues.
DS=$1; S=$2; HO=${3:-}
cd ~/arcane
mkdir -p logs/comp res/comp
TAG=$DS; [ -n "$HO" ] && TAG=${DS}_ho-${HO}
L=logs/comp/${TAG}_seed${S}
run () { local env=$1; shift; local name=$(echo "$*" | tr ' /' '__')
         echo "== $env: $*"
         ( source $env/bin/activate && export PYTHONNOUSERSITE=1 && python "$@" ) > ${L}_${name}.log 2>&1 \
           || echo "FAILED: $* (see ${L}_${name}.log)"; }
# main env (scvi-tools 1.3.3 + harmonypy)
run env comp_scanvi_em.py $DS $S $HO
run env comp_conformal_sc.py $DS $S $HO --features scvi
run env comp_conformal_sc.py $DS $S $HO --features harmony
# env_comp (scArches, symphonypy, scHPL)
run env_comp comp_schpl.py $DS $S $HO
run env_comp comp_symphony.py $DS $S $HO
if [ -z "$HO" ]; then for SP in full cf0 cf1; do run env_comp comp_scpoli.py $DS $S --split $SP; done; fi
# ARCANE on the (scVI, scPoli) pair (needs the three scpoli npz files)
if [ -f emb/${TAG}_scpoli_seed${S}_cf1.npz ]; then
  ( source env/bin/activate && python evaluate2.py ${TAG}_scpoli $S ) > ${L}_arcane_scpoli.log 2>&1 || echo "FAILED: ARCANE scPoli"
fi
# env_popv
run env_popv comp_popv.py $DS $S $HO
( source env/bin/activate && python comp_eval.py $TAG $S ) > ${L}_eval.log 2>&1
tail -40 ${L}_eval.log
