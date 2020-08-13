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
	float sum = 0.0;

	// static shared memory
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

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
    if(row < N && col < M)
        a[row*M + col] = a[row*M + col] - b[row*M + col];
}

__global__ void submatrix_gpu(double* A, double* B, int N, int M, int row_start, int row_end, int col_start, int col_end) {
    uint n_rows = row_end - row_start + 1;
    uint n_cols = col_end - col_start + 1;
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= row_start && row <= row_end && col >= col_start && col <= col_end) {
        printf("A(%d) ---> B(%d)    row = %d    col = %d   val = %lf\n", row*M+col, 
              (row-row_start)*n_cols+col-col_start, row, col, A[row*M+col]);
        B[(row-row_start)*n_cols + col - col_start] = A[row*M + col];
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

    cudaMalloc((void**) &d_G, N*M*sizeof(double));
    cudaMalloc((void**) &d_Gt, M*N*sizeof(double));
    cudaMemcpy(d_G, G, N*M*sizeof(double), cudaMemcpyHostToDevice);
    
    double start = seconds();
    
    transpose_gpu<<<grid, block>>>(d_Gt, d_G, N, M); // transpose G in Gt
    cudaDeviceSynchronize();
    cudaMemcpy(Gt, d_Gt, M*N*sizeof(double), cudaMemcpyDeviceToHost);

    if(N < M) {
        transposed = true;
        M = N;
    }

    cout << "\n----- G -----\n";
    display<double>(G, N, M);

    cout << "\n----- Gt -----\n";
    display<double>(Gt, M, N);

    A    = (double *) malloc(M*M*sizeof(double)); // Gt * G
    S    = (double *) malloc(M*M*sizeof(double)); // lower triangular of A
    L    = (double *) malloc(M*M*sizeof(double)); // lower triangular with zero columns dropped
    Lt   = (double *) malloc(M*M*sizeof(double)); // upper triangular with zero rows dropped
    Lt_L = (double *) malloc(M*M*sizeof(double)); // Lt * L
    I    = (double *) malloc(M*M*sizeof(double)); // inverse of Lt * L

    cudaMalloc((void**) &d_A, M*M*sizeof(double));
    cudaMalloc((void**) &d_S, M*M*sizeof(double));
    cudaMalloc((void**) &d_L, M*M*sizeof(double));
    cudaMalloc((void**) &d_Lt, M*M*sizeof(double));
    cudaMalloc((void**) &d_Lt_L, M*M*sizeof(double));
    cudaMalloc((void**) &d_I, M*M*sizeof(double));

    if(transposed)
        product_gpu<<<grid, block>>>(d_G, d_Gt, d_A, N, N, old_M); // // A = G * Gt 
    else
        product_gpu<<<grid, block>>>(d_Gt, d_G, d_A, old_M, old_M, N); // // A = Gt * G 
    cudaDeviceSynchronize();
    cudaMemcpy(A, d_A, M*M*sizeof(double), cudaMemcpyDeviceToHost);
    
    cout << "\n----- A -----\n";
    display<double>(A, M, M);

    
    double stop = seconds();
    cout << "\nMoore-Penrose pseudoinverse calculation time on GPU: " << stop - start << " seconds" << endl;

}
