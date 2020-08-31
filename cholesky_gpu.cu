__global__ void subtract_gpu(double* a, double* b, int N, int M) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    int i = row*M + col;
    if(row < N && col < M)
        a[i] = a[i] - b[i];
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
        int elem = A[k*N + r-1];
        if(elem > 0) {
            A[k*N + r-1] = sqrt(elem);
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

__global__ void submatrix3_gpu(double* A, double* B, int M, int row_start, int row_end, int col_start, int col_end
                               double* A2, double* B2, int M2, int row_start2, int row_end2, int col_start2, int col_end2
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
        B[(row-row_start2)*n_cols2 + col - col_start2] = A[row*M2 + col];
    }

    if(row >= row_start3 && row <= row_end3 && col >= col_start3 && col <= col_end3) {
        B[(row-row_start3)*n_cols3 + col - col_start3] = A[row*M3 + col];
    }
}

int full_rank_cholesky_decomposition_gpu(double* d_L, double* d_A, int N, dim3 block, dim3 grid) {

    int r = 0;
    double* d_a;
    double* d_b;
    double* d_c;
    double* d_d;
    bool return_v = false;
    bool* d_return_v;

    CHECK( cudaMalloc((void**) &d_a, N*sizeof(double)) );
    CHECK( cudaMalloc((void**) &d_b, N*N*sizeof(double)) );
    CHECK( cudaMalloc((void**) &d_c, N*N*sizeof(double)) );
    CHECK( cudaMalloc((void**) &d_d, N*sizeof(double)) );
    CHECK( cudaMalloc(&d_return_v, sizeof(bool)) );

    for(int k = 0; k < N; k++) {
        r = r+1;

        submatrix_gpu<<<grid, block>>>(d_A, d_a, N, N, k, N-1, k, k);
        
        if(r-2 >= 0) {
            submatrix_gpu<<<grid, block>>>(d_L, d_b, N, N, k, N-1, 0, r-2);
            submatrix_gpu<<<grid, block>>>(d_L, d_c, N, N, k, k, 0, r-2);
            product_gpu<<<grid, block>>>(d_b, d_c, d_d, N-k, 1, r-1);
            subtract_gpu<<<grid, block>>>(d_a, d_d, N-k, 1);
        }

        cp_array_to_matrix<<<grid, block>>>(d_L, d_a, N, r, k);   
        cholesky_sqrt_gpu<<<1, 1>>>(d_return_v, d_L, N, r, k);
        cudaMemcpy(&return_v, d_return_v, sizeof(bool), cudaMemcpyDeviceToHost);

        if(return_v)
            cholesky_division_gpu<<<grid, block>>>(d_L, N, r, k);
        else
            r = r-1;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);

    return r;
}