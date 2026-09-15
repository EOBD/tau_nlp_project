#!/bin/bash
# Sweep v2, slimmed to what needs a GPU (~15 min): the trained model only, extraction only. Contextual run saves just
# the new char-level `after` readouts and per-token chunk positions; the isolated-word run (trained_iso) saves all
# readouts. Then E0 (CPU, 4 tasks) scores 27 new site.readouts (9 trained `after`, 18 trained_iso) and appends them to runs/mi/e0, whose pair sample and
# bootstrap resamples they share, so contrasts with the existing results are paired.
# Submit from an account with gpu-research access:
#   bash /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mi_submit_v2.sh
# Outputs: runs/mi/sweep_v2/feats/{trained,trained_iso}/, runs/mi/e0/report.md (regenerated).
set -e
S=/home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts
KEYS=e0.L0.after,e0.L1.after,e0.L2.after,e0.L3.after,e0.out.after,d0.dechunk.after,d0.resid.after,d0.in.after,d0.out.after
KEYS=$KEYS,e0.emb.mean,e0.out.mean,e1.L0.next,m.in.next,m.L4.next,m.out.next,d1.dechunk.next,d1.resid.next
jid=$(sbatch --parsable --job-name=mi_sweep_v2 --time=00:40:00 $S/mi_sweep.sbatch --out runs/mi/sweep_v2 \
      --model trained=runs/pretrain/h300m_he/model.pt --stages extract --readouts after)
echo "sweep v2 (extraction): job $jid"
jid2=$(sbatch --parsable --job-name=mi_e0_v2 --array=0-3 --dependency=afterok:$jid $S/mi_e0.sbatch \
       --sweep runs/mi/sweep_v2 --out runs/mi/e0 --models trained,trained_iso --keys $KEYS)
echo "E0 on the new features: array job $jid2 (starts after $jid)"
