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

Rscript -e "suppressPackageStartupMessages({library(nlmixr2); library(nlmixr2data)}); cat(\"warfarin exists=\", exists(\"warfarin\"), \"\\n\", sep=\"\"); cat(\"warfarin class=\", paste(class(warfarin), collapse=\",\"), \"\\n\", sep=\"\"); cat(\"warfarin names=\", paste(names(warfarin), collapse=\",\"), \"\\n\", sep=\"\"); cat(\"warfarin nrow=\", nrow(warfarin), \"\\n\", sep=\"\"); if (\"DVID\" %in% names(warfarin)) cat(\"warfarin DVID=\", paste(unique(warfarin$DVID), collapse=\",\"), \"\\n\", sep=\"\"); if (\"CMT\" %in% names(warfarin)) cat(\"warfarin CMT=\", paste(unique(warfarin$CMT), collapse=\",\"), \"\\n\", sep=\"\"); cat(\"Oral_1CPT exists=\", exists(\"Oral_1CPT\"), \"\\n\", sep=\"\"); cat(\"Oral_1CPT class=\", paste(class(Oral_1CPT), collapse=\",\"), \"\\n\", sep=\"\"); cat(\"Oral_1CPT names=\", paste(names(Oral_1CPT), collapse=\",\"), \"\\n\", sep=\"\"); cat(\"Oral_1CPT nrow=\", nrow(Oral_1CPT), \"\\n\", sep=\"\"); if (\"SD\" %in% names(Oral_1CPT)) cat(\"Oral_1CPT SD=\", paste(unique(Oral_1CPT$SD), collapse=\",\"), \"\\n\", sep=\"\"); if (\"SS\" %in% names(Oral_1CPT)) cat(\"Oral_1CPT SS=\", paste(unique(Oral_1CPT$SS), collapse=\",\"), \"\\n\", sep=\"\"); if (\"dose\" %in% names(Oral_1CPT)) cat(\"Oral_1CPT dose=\", paste(unique(Oral_1CPT$dose), collapse=\",\"), \"\\n\", sep=\"\"); print(utils::head(warfarin)); print(utils::head(Oral_1CPT))"
