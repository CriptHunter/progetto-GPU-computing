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

/**
 * Random matrix of number between 0 and `range`
 * @param A random matrix
 * @param N rows of `A`
 * @param M columns of `A`
 * @param range upper limit
 */
void random_matrix(double* A, int N, int M, int range) {
    srand(time(NULL));
    for(int i = 0; i < N; i++)
        for(int j = 0; j < M; j++) {
            A[i*M + j] = rand() % range;
            //A[i*M + j] = i*M + j + 10;
        }

}

/**
 * print a matrix to a text file
 * @param A random matrix
 * @param N rows of `A`
 * @param M columns of `A`
 * @param filename name of the text file
 */
void printf_matrix(double* A, int N, int M, const char* filename) {
    FILE *f = fopen(filename, "w");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++)
            fprintf(f, "%lf\t", A[i*M + j]);
        fprintf(f, "\n");
    }
    fclose(f);
}

/**
 * Test the execution of Pseudoinverse on CPU and GPU comparing the results
 * @param N rows of matrix
 * @param M columns of matrix
 */
void test(int N, int M) {
    double* G = (double *) malloc(N*M*sizeof(double)); // start matrix
    double* A = (double *) malloc(M*N*sizeof(double)); // pseudoinverse
    double* x = (double *) malloc(M*sizeof(double)); // variables vector
    double* y = (double *) malloc(N*sizeof(double)); // constant vector

    random_matrix(G, N, M, 100);
    random_matrix(y, N, 1, 100);
    printf_matrix(G, N, M, "matrix.txt");
    printf_matrix(y, N, 1, "constants_vector.txt");

    cout << "Testing execution on matrix " << N << "×" << M << ":" << endl << endl;

    //execute and print from CPU
    double time_cpu = geninv(G, A, N, M);
    cout << "Moore-Penrose pseudoinverse calculation time on CPU: " << time_cpu << " seconds\n";
    printf_matrix(A, M, N, "pseudoinverse_CPU.txt");

    least_square(A, x, y, N, M);
    printf_matrix(x, M, 1, "least_square_CPU.txt");

    //reset pseudoinverse and variables vector before GPU execution
    memset(A, 0, M*N*sizeof(double));
    memset(x, 0, M*sizeof(double));

    //execute and print from GPU
    double time_gpu = geninv_gpu(G, A, N, M);
    cout << "Moore-Penrose pseudoinverse calculation time on GPU: " << time_gpu << " seconds\n";
    printf_matrix(A, M, N, "pseudoinverse_GPU.txt");

    least_square_gpu(A, x, y, N, M);
    printf_matrix(x, M, 1, "least_square_GPU.txt");

    cout << "\nGPU is " << time_cpu / time_gpu << " times faster than CPU" << endl;
    cout << "---------------------------------------------------------------------------" << endl;

    free(G);
    free(A);
    free(x);
    free(y);
}

int main() {
    // square matrix
    test(16, 16);
    test(32, 32);
    test(64, 64);
    test(128, 128);
    test(256, 256);
    test(512, 512);
    // test(1024, 1024);
    // test(2048, 2048);
    // test(4096, 4096);

    // non-square matrix
    test(16, 8);
    test(32, 1);
    test(64, 15);
    test(20, 128);
    test(50, 256);
    test(512, 490);

    return 0;
}

