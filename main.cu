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


void random_matrix(double* G, int N, int M) {
    srand(time(NULL));
    for(int i = 0; i < N; i++)
        for(int j = 0; j < M; j++)
            G[i*M + j] = rand() % 20;
            //mG[i*M + j] = i*M + j;
}

//print a matrix to a txt file
void printf_matrix(double* G, int N, int M, const char* filename) {
    FILE *f = fopen(filename, "w");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++)
            fprintf(f, "%f\t", G[i*M + j]);
        fprintf(f, "\n");
    }
    fclose(f);
}


int main() {
    int N = 300;
    int M = 300;

    double* G = (double *) malloc(N*M*sizeof(double)); // start matrix
    double* Y = (double *) malloc(M*N*sizeof(double)); // pseudoinverse

    random_matrix(G, N, M);
    printf_matrix(G, N, M, "matrix.txt");
    geninv(G, Y, N, M);
    //printf_matrix(Y, M, N, "pseudoinverse_CPU.txt");

    geninv_gpu(G, Y, N, M);

    free(G);
    free(Y);

    return 0;
}