#!/usr/bin/env bash
cd "$HOME/gemma-chat"
pkill -9 -f ssv_corpus_interp 2>/dev/null || true
pkill -9 -f _remote_omp 2>/dev/null || true
pkill -9 -f uvicorn 2>/dev/null || true
# Clear old caches (50K) so re-caching happens with 500K
rm -f persona_runs/dnd_good_scale/sae/ssv_omp_corpus_cache.json
rm -f persona_runs/dnd_evil/sae/ssv_omp_corpus_cache.json
rm -f persona_runs/dnd_lawful/sae/ssv_omp_corpus_cache.json
rm -f persona_runs/dnd_chaotic/sae/ssv_omp_corpus_cache.json
sleep 1
nohup bash scripts/_remote_omp_interp.sh > logs/omp_interp_v5.log 2>&1 &
echo "PID=$!"
sleep 3
tail -5 logs/omp_interp_v5.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
