#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cmath>
#include <cstring>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>
using namespace std;
#include "geninv_cpu.cpp"
#include "geninv_gpu.cu"


int main() {
    int N = 6;
    int M = 4;

    double* G = (double *) malloc(N*M*sizeof(double)); // start matrix
    double* Y = (double *) malloc(M*N*sizeof(double)); // pseudoinverse

    random_matrix(G, N, M);
    // printf_matrix(G, N, M, "matrix.txt");
    // geninv(G, Y, N, M);
    // printf_matrix(Y, M, N, "pseudoinverse_CPU.txt");

    geninv_gpu(G, Y, N, M);

    free(G);
    free(Y);

    return 0;
}