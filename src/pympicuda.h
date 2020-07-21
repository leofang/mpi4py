#ifndef PyMPI_CUDA_H
#define PyMPI_CUDA_H

/*
 * Currently, we know only the following implementations have
 * the compile-time and runtime checks for CUDA support:
 *
 *     1. Open MPI >= 2.0.0
 *     2. MVAPICH2 >= 2.3.4
 *
 */
#if ((defined(MVAPICH2_NUMVERSION) && MVAPICH2_NUMVERSION >= 20304300) \
     || (defined(OMPI_NUMVERSION) && OMPI_NUMVERSION >= 20000))
#include "mpi-ext.h"
#endif

#ifndef MPIX_CUDA_AWARE_SUPPORT
#undef MPIX_Query_cuda_support
#undef PyMPI_HAVE_CUDA_AWARE_SUPPORT
static int PyMPIX_Query_cuda_support(void)
{
    return -1;  // unclear if there is CUDA awareness or not
}
#define MPIX_Query_cuda_support PyMPIX_Query_cuda_support
#else
#define PyMPI_HAVE_CUDA_AWARE_SUPPORT MPIX_CUDA_AWARE_SUPPORT
#endif


#endif /* PyMPI_CUDA_H */
