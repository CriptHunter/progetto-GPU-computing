// CUDA C program to calculate Moore-Penrose inverse matrix

#define BLOCK_SIZE 32
#define INDEX(rows, cols, stride) (rows * stride + cols)


__global__ void subtract_gpu(double* a, double* b, int N, int M) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    int i = row*M + col;
    if(row < N && col < M)
        a[i] = a[i] - b[i];
}

/**
 * transpose matrix
 * @param At Transpose of `A`
 * @param A Input matrix
 * @param N number of rows of `A`
*/
__global__ void transpose_gpu(double *At, double *A, int N, int M) {
	// static shared memory
	__shared__ double tile[BLOCK_SIZE][BLOCK_SIZE];

	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;

	// trasferimento dati dalla global memory alla shared memory
	if (y < N && x < M)
        tile[threadIdx.y][threadIdx.x] = A[INDEX(y, x, M)];

	// thread synchronization
	__syncthreads();

	// offset blocco trasposto
	y = blockIdx.x * blockDim.x + threadIdx.y;
	x = blockIdx.y * blockDim.y + threadIdx.x;

	// controlli invertiti nelle dim riga colonna
	if (y < M && x < N)
        At[y*N + x] = tile[threadIdx.x][threadIdx.y];
}

/**
 * matrix product
 * @param A First matrix
 * @param B Second matrix
 * @param C `A` * `B`
*/
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


/**
 * Extract a submatrix
 * @param B Submatrix of `A`
 * @param A Input matrix
 * @param N rows of `A`
 * @param M columns of `B`
 * @param row_start starting row index (inclusive)
 * @param row_end ending row index (inclusive)
 * @param col_start starting column (inclusive)
 * @param col_end ending column (inclusive)
 */
__global__ void submatrix_gpu(double* A, double* B, int N, int M, int row_start, int row_end, int col_start, int col_end) {
    uint n_cols = col_end - col_start + 1;
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= row_start && row <= row_end && col >= col_start && col <= col_end) {
        B[(row-row_start)*n_cols + col - col_start] = A[row*M + col];
    }
}


/**
 * divide all the element of row `i` by diagonal element of row `i` for both `A` and `I`
 * @param A Input matrix
 * @param I Partial inverse of `A`
 * @param N Number of rows/columns of `A`
 * @param i current row
 */
__global__ void inverse_no_diag_division_gpu(double *A, double *I, int N, int i){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
        
    __shared__ double tile_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double tile_I[BLOCK_SIZE][BLOCK_SIZE];
    double diag;
    
    if (row == i && col != row && row < N && col < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row*N + col];
            tile_I[threadIdx.y][threadIdx.x] = I[row*N + col];
            diag = A[i*N + i];

            A[i*N + col] = tile_A[threadIdx.y][threadIdx.x] / diag;
            I[i*N + col] = tile_I[threadIdx.y][threadIdx.x] / diag;
    }
}

/**
 * divide diagonal element of row `i` of `I` by diagonal element of row `i` of `A`
 * set diagonal elemento of row `i` of A to zero
 * @param A Input matrix
 * @param I Partial inverse of `A`
 * @param N Number of rows/columns of `A`
 * @param i current row
 */
__global__ void inverse_diag_division_gpu(double *A, double *I, int N, int i){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
        
    if(row == 0 && col == 0) {
        I[i*N + i] = I[i*N + i] / A[i*N + i];
        A[i*N + i]  = 0; // set to 0 to obtain identity on A 
    }
    
}

/**
 * Gauss-Jordan elimination for matrix inverse
 * @param A Input matrix
 * @param I Partial inverse of `A`
 * @param N Number of rows/columns of `A`
 * @param i current row
 */
__global__ void inverse_gauss_jordan_gpu(double *A, double *I, int N, int i) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ double tile_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double tile_A2[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double tile_A3[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double tile_I[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double tile_I2[BLOCK_SIZE][BLOCK_SIZE];
    
	if (col < N && row < N){
		if (row != i){
            tile_A[threadIdx.y][threadIdx.x] = A[row*N + col];
            tile_A2[threadIdx.y][threadIdx.x] = A[row*N + i];
            tile_A3[threadIdx.y][threadIdx.x] = A[i*N + col];
            tile_I[threadIdx.y][threadIdx.x] = I[row*N + col];
            tile_I2[threadIdx.y][threadIdx.x] = I[i*N + col];

            tile_I[threadIdx.y][threadIdx.x] -= tile_I2[threadIdx.y][threadIdx.x] * tile_A2[threadIdx.y][threadIdx.x]; 
            
            if (col != i)
                tile_A[threadIdx.y][threadIdx.x] -= tile_A3[threadIdx.y][threadIdx.x] * tile_A2[threadIdx.y][threadIdx.x];
            
            A[row*N + col] = tile_A[threadIdx.y][threadIdx.x];
            I[row*N + col] = tile_I[threadIdx.y][threadIdx.x];
		}
	}
}

/**
 * initialize identity matrix
 * @param A input matrix
 * @param N matrix order
 */
__global__ void inverse_init_identity_gpu(double* A, int N) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < N && col < N) {
        if (row == col) 
            A[row*N + col] = 1.0;
        else 
            A[row*N + col] = 0.0;
    }
}

/**
 * Moore-Penrose generalized inverse matrix
 * @param G Input matrix
 * @param Y Generalized inverse
 * @param N number of rows of `G` 
 * @param M number of columns `G`
 * @return execution time
 */
double geninv_gpu(double* G, double* Y, int N, int M) {
    int old_M = M; // to remember M original value
    bool transposed = false; // true if N < M
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

    // pseudoinverse formula is different if N < M
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
    
    double aaa = seconds();
    cudaMemcpy(A, d_A, M*M*sizeof(double), cudaMemcpyDeviceToHost);
    int rank = full_rank_cholesky_decomposition(A, S, M); // S = cholesky(A)
    cudaMemcpy(d_S, S, M*M*sizeof(double), cudaMemcpyHostToDevice);
    //int rank = full_rank_cholesky_decomposition_gpu(d_S, d_A, M, block, grid);
    double bbb = seconds();
    cout << "S on GPU: " << bbb - aaa << endl;

    submatrix_gpu<<<grid, block>>>(d_S, d_L, M, M, 0, M, 0, rank-1); // S = L with zeros columns dropped
    transpose_gpu<<<grid, block>>>(d_Lt, d_L, M, rank); // transpose of L
    product_gpu<<<grid, block>>>(d_Lt, d_L, d_Lt_L, rank, rank, M); // Lt_L = Lt * L


    aaa = seconds();
    // I = inv(Lt_L)
    inverse_init_identity_gpu<<<grid, block>>>(d_I, rank);
    for (int i = 0; i < rank; i++){
        inverse_no_diag_division_gpu <<<grid, block>>>(d_Lt_L, d_I, rank, i);
        inverse_diag_division_gpu <<<1, 1>>>(d_Lt_L, d_I, rank, i);
        inverse_gauss_jordan_gpu <<<grid, block>>>(d_Lt_L, d_I, rank, i);
    }
    bbb = seconds();
    cout << "inverse on GPU: " << bbb - aaa << endl;

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

    return stop - start;
}
