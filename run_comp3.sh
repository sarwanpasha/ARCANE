#!/bin/bash
# Competitors (all except popV, which runs elsewhere) for one dataset/seed[/holdout], then ARCANE on (scVI, scPoli)
DS=$1; S=$2; HO=${3:-}
cd ~/arcane
TAG=$DS; [ -n "$HO" ] && TAG=${DS}_ho-${HO}
L=logs/comp/${TAG}_seed${S}
run () { local env=$1; shift; local name=$(echo "$*" | tr ' /' '__')
         ( source $env/bin/activate && export PYTHONNOUSERSITE=1 && python "$@" ) > ${L}_${name}.log 2>&1 || echo "FAILED: $*"; }
run env comp_scanvi_em.py $DS $S $HO
run env comp_conformal_sc.py $DS $S $HO --features scvi
run env comp_conformal_sc.py $DS $S $HO --features harmony
run env_comp comp_schpl.py $DS $S $HO
run env_comp comp_symphony.py $DS $S $HO
for SP in full cf0 cf1; do run env_comp comp_scpoli.py $DS $S $HO --split $SP; done
if [ -f emb/${TAG}_scpoli_seed${S}_cf1.npz ]; then
  ( source env/bin/activate && python evaluate2.py ${TAG}_scpoli $S ) > ${L}_arcane_scpoli.log 2>&1 || echo "FAILED: ARCANE scPoli"
fi
run env_comp symphony_views.py $DS $S $HO
if [ -f emb/${TAG}_symph_seed${S}_cf1.npz ]; then
  ( source env/bin/activate && python evaluate2.py ${TAG}_symph $S ) > ${L}_arcane_symph.log 2>&1 || echo "FAILED: ARCANE Symphony"
fi
( source env/bin/activate && python comp_eval.py $TAG $S ) > ${L}_eval.log 2>&1
echo COMP3_DONE $TAG $S
