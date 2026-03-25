#!/bin/bash -l
#SBATCH --partition=cpu
#SBATCH --account=lilly-agentic-cpu
#SBATCH --qos=standby
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# Ensure module command exists in non-login contexts.
if ! type module >/dev/null 2>&1; then
  for __mod_init in /etc/profile /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash; do
    if [ -r "$__mod_init" ]; then
      . "$__mod_init" >/dev/null 2>&1 || true
      if type module >/dev/null 2>&1; then
        break
      fi
    fi
  done
fi
if ! type module >/dev/null 2>&1; then
  echo 'cluster-slurm: module command unavailable; adjust profile.module_init_scripts' >&2
  exit 127
fi

module --force purge
module use /scratch/gautschi/$USER/codex-modules
module load conda-env/nlmixr2-ref-py3.12.11
export PATH=/scratch/gautschi/$USER/codex-envs/nlmixr2-ref/bin:$PATH

Rscript reference_warfarin_saem_fit.R
