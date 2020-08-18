#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cmath>
#include <cstring>
#include <time.h>
#include <stdio.h>
#include "common.h"
using namespace std;
#include "geninv_cpu.cpp"
#include "geninv_gpu.cu"


void random_matrix(double* G, int N, int M) {
    srand(time(NULL));
    for(int i = 0; i < N; i++)
        for(int j = 0; j < M; j++)
            G[i*M + j] = rand() % 10;
            //G[i*M + j] = i*M + j + 10;
}

//print a matrix to a txt file
void printf_matrix(double* A, int N, int M, const char* filename) {
    FILE *f = fopen(filename, "w");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++)
            fprintf(f, "%lf\t", A[i*M + j]);
        fprintf(f, "\n");
    }
    fclose(f);
}

int main() {
    int N = 1500;
    int M = 1500;

    double* G = (double *) malloc(N*M*sizeof(double)); // start matrix
    double* Y = (double *) malloc(M*N*sizeof(double)); // pseudoinverse

    random_matrix(G, N, M);
    printf_matrix(G, N, M, "matrix.txt");

    //exec and print from CPU
    double time_cpu = geninv(G, Y, N, M);
    cout << "\nMoore-Penrose pseudoinverse calculation time on CPU: " << time_cpu << " seconds" << endl;
    printf_matrix(Y, M, N, "pseudoinverse_CPU.txt");

    memset(Y, 0, M*N*sizeof(double));

    //exec and print from GPU
    double time_gpu = geninv_gpu(G, Y, N, M);
    cout << "\nMoore-Penrose pseudoinverse calculation time on GPU: " << time_gpu << " seconds" << endl;
    printf_matrix(Y, M, N, "pseudoinverse_GPU.txt");

    cout << "\n GPU is " << time_cpu / time_gpu << " times faster than CPU" << endl;

    free(G);
    free(Y);

    return 0;
}