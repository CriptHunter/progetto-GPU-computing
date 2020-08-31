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

void random_matrix(double* A, int N, int M) {
    srand(time(NULL));
    for(int i = 0; i < N; i++)
        for(int j = 0; j < M; j++)
            A[i*M + j] = rand() % 10;
            //A[i*M + j] = i*M + j + 10;
}

__global__ void cp_array_to_matrix(double* A, double* B, int N, int r, int k) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    uint i = row * N + col + k;

    if(i < N)
        A[i*N + r-1] = B[i-k];    
}


__global__ void cholesky_sqrt_gpu(bool* return_v, double* A, int N, int r, int k) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row == 0 && col == 0) {
        if(A[k*N + r-1] > 0) {
            A[k*N + r-1] = sqrt(A[k*N + r-1]);
            *return_v = true;
        }
        else 
            *return_v = false;
    }
}

__global__ void cholesky_division_gpu(double* A, int N, int r, int k) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    uint i = row * N + col + k+1;

    if(i < N) {
        A[i*N + r-1] = A[i*N + r-1] / A[k*N + r-1]; 
    }
}

__global__ void subtract_gpu(double* a, double* b, int N, int M) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row*M + col;

    if(row < N && col < M) {
        if(a[i] - b[i] < 1E-7 && a[i] - b[i] >= 0)
            a[i] = 0;
        else
            a[i] = a[i] - b[i];
      }

}


__global__ void submatrix3_gpu(double* A, double* B, int M, int row_start, int row_end, int col_start, int col_end,
    double* A2, double* B2, int M2, int row_start2, int row_end2, int col_start2, int col_end2,
    double* A3, double* B3, int M3, int row_start3, int row_end3, int col_start3, int col_end3) {
    uint n_cols = col_end - col_start + 1;
    uint n_cols2 = col_end2 - col_start2 + 1;
    uint n_cols3 = col_end3 - col_start3 + 1;
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= row_start && row <= row_end && col >= col_start && col <= col_end) {
        B[(row-row_start)*n_cols + col - col_start] = A[row*M + col];
    }

    if(row >= row_start2 && row <= row_end2 && col >= col_start2 && col <= col_end2) {
        B2[(row-row_start2)*n_cols2 + col - col_start2] = A2[row*M2 + col];
    }

    if(row >= row_start3 && row <= row_end3 && col >= col_start3 && col <= col_end3) {
        B3[(row-row_start3)*n_cols3 + col - col_start3] = A3[row*M3 + col];
    }
}


int full_rank_cholesky_decomposition_gpu(double* d_L, double* d_A, int N, dim3 block, dim3 grid) {

    int r = 0;
    double* d_a;
    double* d_b;
    double* d_c;
    double* d_d;

    double* a = (double*) malloc(N*sizeof(double));
    double* L = (double*) malloc(N*N*sizeof(double));
    memset(L, 0, N*N*sizeof(double));


    CHECK( cudaMalloc((void**) &d_a, N*sizeof(double)) );
    CHECK( cudaMalloc((void**) &d_b, N*N*sizeof(double)) );
    CHECK( cudaMalloc((void**) &d_c, N*N*sizeof(double)) );
    CHECK( cudaMalloc((void**) &d_d, N*sizeof(double)) );
    

    for(int k = 0; k < N; k++) {
        r = r+1;

        if(r < 2)
            submatrix_gpu<<<grid, block>>>(d_A, d_a, N, N, k, N-1, k, k);
        
        else {
            // submatrix_gpu<<<grid, block>>>(d_A, d_a, N, N, k, N-1, k, k);
            // submatrix_gpu<<<grid, block>>>(d_L, d_b, N, N, k, N-1, 0, r-2);
            // submatrix_gpu<<<grid, block>>>(d_L, d_c, N, N, k, k, 0, r-2);
            submatrix3_gpu<<<grid, block>>>(d_A, d_a, N, k, N-1, k, k,
                                            d_L, d_b, N, k, N-1, 0, r-2,
                                            d_L, d_c, N, k, k, 0, r-2);
            product_gpu<<<grid, block>>>(d_b, d_c, d_d, N-k, 1, r-1);
            subtract_gpu<<<grid, block>>>(d_a, d_d, N-k, 1);
        }

        cudaMemcpy(a, d_a, N*sizeof(double), cudaMemcpyDeviceToHost);

        for(int i = k; i < N; i++)
            L[i*N + r-1] = a[i-k];
        
        if(L[k*N + r-1] > 5E-9) {
            L[k*N + r-1] = sqrt(L[k*N + r-1]);

            if (k+1 < N)
                for(int i = k+1; i < N; i++)
                    L[i*N + r-1] = L[i*N + r-1] / L[k*N + r-1]; 
        }
        else
            r = r-1;

        cudaMemcpy(d_L, L, N*N*sizeof(double), cudaMemcpyHostToDevice);

    }

    return r;
}

__global__ void empty() {
    return;
}

int main() {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    int N = 3000;
    int M = 10;
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y, 1);

    // double* A = (double *) malloc(N*N*sizeof(double));
    // double* S = (double *) malloc(N*N*sizeof(double));
    // double* d_A;
    // CHECK( cudaMalloc((void**) &d_A, N*N*sizeof(double)) );
    // double* d_S;
    // CHECK( cudaMalloc((void**) &d_S, N*N*sizeof(double)) );

    // random_matrix(A, N, N);

    // double start = seconds();
    // int rank = full_rank_cholesky_decomposition(A, S, N);
    // double stop = seconds();
    // cout << "tempo cpu: " << stop - start << endl;

    // //display<double>(S, N, N);
    // memset(S, 0, N*N*sizeof(double));
    // cout << "-----------------------------------------------------------------------------------------------\n";

    // cudaMemcpy(d_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
    // start = seconds();
    // full_rank_cholesky_decomposition_gpu(d_S, d_A, N, block, grid);
    // cudaDeviceSynchronize();
    // stop = seconds();
    // cout << "tempo gpu: " << stop - start << endl;

    // cudaMemcpy(S, d_S, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    // //display<double>(S, N, N);

    double start = seconds();
    empty<<<1, 1>>>();
    //cudaDeviceSynchronize();
    double stop = seconds();
    cout << "tempo gpu: " << stop - start << endl;


}