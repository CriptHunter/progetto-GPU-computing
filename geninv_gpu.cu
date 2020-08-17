// Dimensione del blocco
#define BLOCK_SIZE 32
// macro x conversione indici lineari
#define INDEX(rows, cols, stride) (rows * stride + cols)

__global__ void transpose_gpu(double *a_t, double *a, int nrows, int ncols) {
	// static shared memory
	__shared__ double tile[BLOCK_SIZE][BLOCK_SIZE];

	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;

	// trasferimento dati dalla global memory alla shared memory
	if (y < nrows && x < ncols)
        tile[threadIdx.y][threadIdx.x] = a[INDEX(y, x, ncols)];

	// thread synchronization
	__syncthreads();

	// offset blocco trasposto
	y = blockIdx.x * blockDim.x + threadIdx.y;
	x = blockIdx.y * blockDim.y + threadIdx.x;

	// controlli invertiti nelle dim riga colonna
	if (y < ncols && x < nrows)
        a_t[y*nrows + x] = tile[threadIdx.x][threadIdx.y];
}

__global__ void product_gpu(double* A, double* B, double* C, int N, int M, int P) {
	// indexes
	uint row = blockIdx.y * blockDim.y + threadIdx.y;
	uint col = blockIdx.x * blockDim.x + threadIdx.x;

	// target: compute the right sum for the given row and col
	double sum = 0.0;

	// static shared memory
	__shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

	// loop over blocks from block row of matrix A
	// and block column of matrix B
	uint numBlocks = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
	for (uint m = 0; m < numBlocks; m++) {

		// copy block from matrix to shared memory
		uint r = m * BLOCK_SIZE + threadIdx.y;
		uint c = m * BLOCK_SIZE + threadIdx.x;
		As[threadIdx.y][threadIdx.x] = A[INDEX(row, c, P)];
		Bs[threadIdx.y][threadIdx.x] = B[INDEX(r, col, M)];

		//---------------------------------------------------------------
		__syncthreads();  //  BARRIER SYNC on SMEM loading

		// length of this part of row-column product is BLOCK_SIZE
		// except for last block when it may be smaller
		uint K = BLOCK_SIZE;
		if (m == numBlocks - 1) K = P - m * BLOCK_SIZE; // tune last block

		// compute this part of row-column product
		for (uint k = 0; k < K; k++)
			sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

		//---------------------------------------------------------------
		__syncthreads();  //  BARRIER SYNC on prod over blocks
		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
	}

	// store computed element in matrix C
	if (row < N && col < M)
		C[row * M + col] = sum;
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

__global__ void submatrix_gpu(double* A, double* B, int N, int M, int row_start, int row_end, int col_start, int col_end) {
    uint n_cols = col_end - col_start + 1;
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= row_start && row <= row_end && col >= col_start && col <= col_end) {
        B[(row-row_start)*n_cols + col - col_start] = A[row*M + col];
    }

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

__global__ void drop_zero_column_gpu(double* B, double* A, int N, int rank) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < rank) {
        B[row*N + col] = A[row*N + col];
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

__global__ void nodiag_normalize(double *A, double *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == i && x!=y){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}
	
}

__global__ void diag_normalize(double *A, double *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == y && x == i){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}

}

__global__ void gaussjordan(double *A, double *I, int n, int i) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			I[x*n + y] -= I[i*n + y] * A[x*n + i];
			if (y != i){
				A[x*n + y] -= A[i*n + y] * A[x*n + i];
			}	 
		}
	}
}

__global__ void set_zero(double *A, double *I, int n, int i) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			if (y == i){
				A[x*n + y] = 0;
			}
		}
	}
}

__global__ void set_diagonal(double* A, int N) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < N && col < N) {
        if (row == col) 
            A[row*N + col] = 1.0;
        else 
            A[row*N + col] = 0.0;
    }
}

void geninv_gpu(double* G, double* Y, int N, int M) {
    int old_M = M;
    bool transposed = false;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y, 1);
    

    //cpu variables declaration
    double* Gt = (double *) malloc(M*N*sizeof(double)); // transpose of G
    double* A; // Gt * G
    double* S; // lower triangular of A
    double* L; // lower triangular with zero columns dropped
    double* Lt; // upper triangular with zero rows dropped
    double* Lt_L; // Lt * L
    double* I; // inverse of Lt * L

    //gpu variables declaration
    double* d_G;
    double* d_Gt;
    double* d_A;
    double* d_S;
    double* d_L;
    double* d_Lt;
    double* d_Lt_L;
    double* d_I;
    double* d_Y;

    CHECK( cudaMalloc((void**) &d_G, N*M*sizeof(double)) );
    CHECK( cudaMalloc((void**) &d_Gt, M*N*sizeof(double)) );
    cudaMemcpy(d_G, G, N*M*sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    double start = seconds();
    
    transpose_gpu<<<grid, block>>>(d_Gt, d_G, N, M); // transpose G in Gt
    //cudaMemcpy(Gt, d_Gt, M*N*sizeof(double), cudaMemcpyDeviceToHost);

    if(N < M) {
        transposed = true;
        M = N;
    }

    //cout << "\n----- G -----\n";
    //display<double>(G, N, M);

    // cout << "\n----- Gt -----\n";
    // display<double>(Gt, M, N);

    cudaFree(G);
    //cudaFree(Gt);

    A    = (double *) malloc(M*M*sizeof(double)); // Gt * G
    S    = (double *) malloc(M*M*sizeof(double)); // lower triangular of A
    L    = (double *) malloc(M*M*sizeof(double)); // lower triangular with zero columns dropped
    Lt   = (double *) malloc(M*M*sizeof(double)); // upper triangular with zero rows dropped
    Lt_L = (double *) malloc(M*M*sizeof(double)); // Lt * L
    I    = (double *) malloc(M*M*sizeof(double)); // inverse of Lt * L

    CHECK( cudaMalloc((void**) &d_A, M*M*sizeof(double)) );
    CHECK( cudaMalloc((void**) &d_S, M*M*sizeof(double)) );
    CHECK( cudaMalloc((void**) &d_L, M*M*sizeof(double)) );
    CHECK( cudaMalloc((void**) &d_Lt, M*M*sizeof(double)) );
    CHECK( cudaMalloc((void**) &d_Lt_L, M*M*sizeof(double)) );
    CHECK( cudaMalloc((void**) &d_I, M*M*sizeof(double)) );

    if(transposed)
        product_gpu<<<grid, block>>>(d_G, d_Gt, d_A, N, N, old_M); // // A = G * Gt 
    else
        product_gpu<<<grid, block>>>(d_Gt, d_G, d_A, old_M, old_M, N); // // A = Gt * G 

    cudaMemcpy(A, d_A, M*M*sizeof(double), cudaMemcpyDeviceToHost);
    // cout << "\n----- A -----\n";
    // display<double>(A, M, M);

    //cout << "\n----- S -----\n";
    //int rank = full_rank_cholesky_decomposition_gpu(d_S, d_A, M, block, grid);
    int rank = full_rank_cholesky_decomposition(A, S, M); // S = cholesky(A)
    //display(S, M, M);
    cudaMemcpy(d_S, S, M*M*sizeof(double), cudaMemcpyHostToDevice);

    // cudaMemcpy(S, d_S, M*M*sizeof(double), cudaMemcpyDeviceToHost);
    // display<double>(S, M, M);

    //cout << "\n----- L (" << M - rank << " columns dropped)-----\n";
    submatrix_gpu<<<grid, block>>>(d_S, d_L, M, M, 0, M, 0, rank-1);
    // cudaMemcpy(L, d_L, M*M*sizeof(double), cudaMemcpyDeviceToHost);
    // display<double>(L, M, rank);

    // cout << "\n----- Lt -----\n";
    transpose_gpu<<<grid, block>>>(d_Lt, d_L, M, rank);
    // cudaMemcpy(Lt, d_Lt, rank*M*sizeof(double), cudaMemcpyDeviceToHost);
    // display<double>(Lt, rank, M);

    // cout << "\n----- Lt * L -----\n";
    product_gpu<<<grid, block>>>(d_Lt, d_L, d_Lt_L, rank, rank, M);
    // cudaMemcpy(Lt_L, d_Lt_L, M*M*sizeof(double), cudaMemcpyDeviceToHost);
    // display<double>(Lt_L, rank, rank);

    //cout << "\n----- I -----\n";
    set_diagonal<<<grid, block>>>(d_I, rank);

    for (int i = 0; i < rank; i++){
		nodiag_normalize <<<grid, block>>>(d_Lt_L, d_I, rank, i);
		diag_normalize <<<grid, block>>>(d_Lt_L, d_I, rank, i);
		gaussjordan <<<grid, block>>>(d_Lt_L, d_I, rank, i);
		set_zero <<<grid, block>>>(d_Lt_L, d_I, rank, i);
    }
    
    cudaDeviceSynchronize();
    // cudaMemcpy(I, d_I, rank*rank*sizeof(double), cudaMemcpyDeviceToHost);
    // display<double>(I, rank, rank);

    double* d_tmp;
    double* d_tmp1; 
    double* d_tmp2;

    if(transposed) { // Y = Gt * L * I * I * Lt
        // tmp =  (double *) malloc(old_M*rank*sizeof(double));
        // tmp1 = (double *) malloc(old_M*rank*sizeof(double));
        // tmp2 = (double *) malloc(old_M*rank*sizeof(double));

        // multiply(Gt, old_M, N, L, N, rank, tmp);
        // multiply(tmp, old_M, rank, I, rank, rank, tmp1);
        // multiply(tmp1, old_M, rank, I, rank, rank, tmp2);
        // multiply(tmp2, old_M, rank, Lt, rank, N, Y);
    }
    else { // Y = L * I * I * Lt * Gt
        CHECK( cudaMalloc((void**) &d_tmp, old_M*rank*sizeof(double)) );
        CHECK( cudaMalloc((void**) &d_tmp1, old_M*rank*sizeof(double)) );
        CHECK( cudaMalloc((void**) &d_tmp2, old_M*old_M*sizeof(double)) );

        product_gpu<<<grid, block>>>(d_L, d_I, d_tmp, M, rank, rank);
        product_gpu<<<grid, block>>>(d_tmp, d_I, d_tmp1, M, rank, rank);
        product_gpu<<<grid, block>>>(d_tmp1, d_Lt, d_tmp2, M, M, rank);
        CHECK( cudaMalloc((void**) &d_Y, M*N*sizeof(double)) );
        product_gpu<<<grid, block>>>(d_tmp2, d_Gt, d_Y, M, N, M);

        // multiply(L, M, rank, I, rank, rank, tmp);
        // multiply(tmp, M, rank, I, rank, rank, tmp1);
        // multiply(tmp1, M, rank, Lt, rank, M, tmp2); 
        // multiply(tmp2, M, M, Gt, M, N, Y);
    }

    cudaDeviceSynchronize();
    double stop = seconds();

    CHECK( cudaMemcpy(Y, d_Y, old_M*N*sizeof(double), cudaMemcpyDeviceToHost) );
    cudaDeviceSynchronize();
    //display<double>(Y, old_M, N);

    // free(Gt);
    // free(A);
    // free(I);
    // free(S);
    // free(L);
    // free(Lt);
    // free(Lt_L);

    //cudaFree(d_Gt);
    cudaFree(d_A);
    cudaFree(d_I);
    cudaFree(d_S);
    cudaFree(d_L);
    cudaFree(d_Lt);
    cudaFree(d_Lt_L);
    cudaFree(d_Y);
    cudaDeviceReset();


    cout << "\nMoore-Penrose pseudoinverse calculation time on GPU: " << stop - start << " seconds" << endl;
}
