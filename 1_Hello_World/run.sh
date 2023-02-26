#!/bin/sh
#PJM -L rscgrp=regular
#PJM -L node=2
#PJM --mpi proc=20
#PJM -L elapse=0:05:00
#PJM -g XXXXX
#PJM -j

mpiexec.hydra -n ${PJM_MPI_PROC} ./a.out
