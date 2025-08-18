#!/bin/bash

## USAGE: 
## Working directory: cd directory/for/this/project
## Starting predictions: sh example.prediction.sh


PROJECT="../"
cd ${PROJECT}

CODEDIR="${PROJECT}/code"
DATADIR="${PROJECT}/results"
mkdir -p ${DATADIR}/logs

POLYAID="${PROJECT}/resources/published_models/PolyaID.h5"
POLYASTR="${PROJECT}/resources/published_models/PolyaStrength.h5"

DATA="${DATADIR}/example_sequences.ar.txt"
DNAME="example_ar"


## launch dataset prediction jobs

> ${DATADIR}/logs/example_prediction.${DNAME}.outlog
> ${DATADIR}/logs/example_prediction.${DNAME}.errlog

cat > ${DATADIR}/logs/example_prediction.${DNAME}.launch.pbs  <<- EOM
#!/bin/bash

#PBS -A b1042
#PBS -p genomics
#PBS -J "example_pred"
#PBS -t 00:05:00
#PBS -N 1
#PBS -n 1
#PBS --mem 2G
#PBS--output=${DATADIR}/logs/example_prediction.${DNAME}.outlog
#PBS --error=${DATADIR}/logs/example_prediction.${DNAME}.errlog


cd ${DATADIR}
module load python/anaconda
source activate
conda activate NAMEOFENV(Your_conda_env)

echo -e "\nBeginning predictions for ${DNAME}"

python ${CODEDIR}/example.prediction.py --model ${POLYAID} --modeltype "polyaid" \
 --data ${DATA} --dataname ${DNAME} --outdir ${DATADIR} && echo -e "\tCompleted successfully for PolyaID"

python ${CODEDIR}/example.prediction.py --model ${POLYASTR} --modeltype "polyastrength" \
 --data ${DATA} --dataname ${DNAME} --outdir ${DATADIR} && echo -e "\tCompleted successfully for PolyaStrength"

EOM

echo "Submitting job for "example_prediction.${DNAME}.launch.pbs
# ${DATADIR}/logs/example_prediction.${DNAME}.launch.pbs
sh ${DATADIR}/logs/example_prediction.${DNAME}.launch.pbs




