#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Hello, I'm " << rank << " / " << size << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
