#define BLOCK_SIZE 32
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

__global__ void submatrix_gpu(double* A, double* B, int N, int M, int row_start, int row_end, int col_start, int col_end) {
    uint n_cols = col_end - col_start + 1;
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= row_start && row <= row_end && col >= col_start && col <= col_end) {
        B[(row-row_start)*n_cols + col - col_start] = A[row*M + col];
    }
}

__global__ void nodiag_normalize(double *A, double *I, int N, int i){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < N && row < N)
	if (col == i && col!=row){
		I[col*N + row] /= A[i*N + i];
		A[col*N + row] /= A[i*N + i];
	}
	
}

__global__ void diag_normalize(double *A, double *I, int N, int i){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < N && row < N)
	if (col == row && col == i){
		I[col*N + row] /= A[i*N + i];
		A[col*N + row] /= A[i*N + i];
	}

}

__global__ void gaussjordan(double *A, double *I, int N, int i) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < N && row < N){
		if (col != i){
			I[col*N + row] -= I[i*N + row] * A[col*N + i];
			if (row != i){
				A[col*N + row] -= A[i*N + row] * A[col*N + i];
			}	 
		}
	}
}

__global__ void set_zero(double *A, double *I, int N, int i) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < N && row < N){
		if (col != i){
			if (row == i){
				A[col*N + row] = 0;
			}
		}
	}
}

__global__ void init_identity_gpu(double* A, int N) {
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
    int old_M = M; //to remember M original value
    bool transposed = false; //true if N < M
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y, 1);
    
    //cpu variables
    double* A;
    double* S;

    //gpu variables
    double* d_G;     //original matrix
    double* d_Gt;    // transpose of G
    double* d_A;     // Gt * G
    double* d_S;     // lower triangular of A
    double* d_L;     // lower triangular with zero columns dropped
    double* d_Lt;    // upper triangular with zero rows dropped
    double* d_Lt_L;  // Lt * L
    double* d_I;     // inverse of Lt * L
    double* d_Y;     //pseudoinverse of G

    CHECK( cudaMalloc((void**) &d_G, N*M*sizeof(double)) );
    CHECK( cudaMalloc((void**) &d_Gt, M*N*sizeof(double)) );
    cudaMemcpy(d_G, G, N*M*sizeof(double), cudaMemcpyHostToDevice);
    
    double start = seconds();
    
    transpose_gpu<<<grid, block>>>(d_Gt, d_G, N, M); // transpose G in Gt

    //pseudoinverse formula is different if N < M
    if(N < M)  {
        transposed = true;
        M = N;
    }

    cudaFree(G);

    A = (double*) malloc(M*M*sizeof(double));
    S = (double*) malloc(M*M*sizeof(double));
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
    int rank = full_rank_cholesky_decomposition(A, S, M); // S = cholesky(A)
    cudaMemcpy(d_S, S, M*M*sizeof(double), cudaMemcpyHostToDevice);

    submatrix_gpu<<<grid, block>>>(d_S, d_L, M, M, 0, M, 0, rank-1); // S = L with 0 columns dropped
    transpose_gpu<<<grid, block>>>(d_Lt, d_L, M, rank); // transpose of L
    product_gpu<<<grid, block>>>(d_Lt, d_L, d_Lt_L, rank, rank, M); // Lt_L = Lt * L

    // I = inv(Lt_L)
    init_identity_gpu<<<grid, block>>>(d_I, rank);
    for (int i = 0; i < rank; i++){
		nodiag_normalize <<<grid, block>>>(d_Lt_L, d_I, rank, i);
		diag_normalize <<<grid, block>>>(d_Lt_L, d_I, rank, i);
		gaussjordan <<<grid, block>>>(d_Lt_L, d_I, rank, i);
		set_zero <<<grid, block>>>(d_Lt_L, d_I, rank, i);
    }

    double* d_tmp;
    double* d_tmp1; 
    double* d_tmp2;

    if(transposed) { // Y = Gt * L * I * I * Lt
        CHECK( cudaMalloc((void**) &d_tmp, old_M*rank*sizeof(double)) );
        CHECK( cudaMalloc((void**) &d_tmp1, old_M*rank*sizeof(double)) );
        CHECK( cudaMalloc((void**) &d_tmp2, old_M*rank*sizeof(double)) );

        product_gpu<<<grid, block>>>(d_Gt, d_L, d_tmp, old_M, rank, N);
        product_gpu<<<grid, block>>>(d_tmp, d_I, d_tmp1, old_M, rank, rank);
        product_gpu<<<grid, block>>>(d_tmp1, d_I, d_tmp2, old_M, rank, rank);
        CHECK( cudaMalloc((void**) &d_Y, old_M*N*sizeof(double)) );
        product_gpu<<<grid, block>>>(d_tmp2, d_Lt, d_Y, old_M, N, rank);
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
    }

    cudaDeviceSynchronize();
    double stop = seconds();

    CHECK( cudaMemcpy(Y, d_Y, old_M*N*sizeof(double), cudaMemcpyDeviceToHost) );
    cudaDeviceSynchronize();

    free(A);
    free(S);
    cudaFree(d_Gt);
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
