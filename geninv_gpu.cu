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
        //printf("A(%d) ---> B(%d)    row = %d    col = %d   val = %lf\n", row*M+col, 
                //(row-row_start)*n_cols+col-col_start, row, col, A[row*M+col]);
        B[(row-row_start)*n_cols + col - col_start] = A[row*M + col];
    }
}

//calculate tollerance using diagonal of square matrix
__global__ void tollerance(double*A, int N, double* tol) {
    *tol = 900; // TODO!
}

__global__ void cp_array_to_matrix(double* A, double* B, int N, int r, int k) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    uint i = row * N + col + k;

    if(row == 0 && col == 0)
    printf("on gpu: %lf", B[i-k]);

    if(i < N)
        A[i*N + r-1] = B[i-k];    
    

}

__global__ void drop_zero_column_gpu(double* B, double* A, int N, int rank) {
    // int k = 0;
    // for(int i = 0; i < n; i++)
    //     for(int j = 0; j < rank; j++) {
    //         b[k] = a[i*n + j];
    //         k++;
    //     }   

    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < rank) {
        B[row*N + col] = A[row*N + col];
    }
}

int full_rank_cholesky_decomposition_gpu(double* A, double* L, int N, dim3 block, dim3 grid) {
    double* h_L = (double*) malloc(N*N*sizeof(double));
    // double tol = A[0];

    // for(int i = 0; i < N; i++) {
    //     double k = A[i*N + i];
    //     if(k < tol)
    //         tol = k;
    // }
    double tol = 50 * 1E-9;

    // double tol;
    // double* d_tol;
    // cudaMalloc(&d_tol, sizeof(double));
    // tollerance<<<grid,block>>>(A, N, d_tol);
    // cudaMemcpy(&tol, d_tol, sizeof(double), cudaMemcpyDeviceToHost); 

    int r = 0;

    for(int k = 0; k < N; k++) {
        r = r+1;

        double* a = (double*) malloc((N-k)*sizeof(double));
        double* d_a;
        CHECK( cudaMalloc((void**) &d_a, (N-k)*sizeof(double)) );
        submatrix_gpu<<<grid, block>>>(A, d_a, N, N, k, N-1, k, k);
        //cudaMemcpy(a, d_a, (N-k)*sizeof(double), cudaMemcpyDeviceToHost);

        // cout << "matrix A: " << endl;
        // display(a, N-k, 1);
        // cout << endl;


        if(r-2 >= 0) {
            //double* b = (double*) malloc((N-k)*(r-1)*sizeof(double));
            double* d_b;
            CHECK( cudaMalloc((void**) &d_b, (N-k)*(r-1)*sizeof(double)) );
            submatrix_gpu<<<grid, block>>>(L, d_b, N, N, k, N-1, 0, r-2);
            cudaDeviceSynchronize();
            //cudaMemcpy(b, d_b, (N-k)*(r-1)*sizeof(double), cudaMemcpyDeviceToHost);

            //double* c = (double*) malloc((r-1)*sizeof(double));
            double* d_c;
            CHECK( cudaMalloc((void**) &d_c, (r-1)*sizeof(double)) );
            submatrix_gpu<<<grid, block>>>(L, d_c, N, N, k, k, 0, r-2);
            cudaDeviceSynchronize();
            //cudaMemcpy(c, d_c, (r-1)*sizeof(double), cudaMemcpyDeviceToHost);

            //double* ct = (double*) malloc((r-1)*sizeof(double));
            double* d_ct;
            CHECK( cudaMalloc((void**) &d_ct, (r-1)*sizeof(double)) );
            transpose_gpu<<<grid, block>>>(d_ct, d_c, 1, r-1);
            cudaDeviceSynchronize();
            //cudaMemcpy(ct, d_ct, (r-1)*sizeof(double), cudaMemcpyDeviceToHost);

            //double* d = (double*) malloc((N-k)*sizeof(double));
            double* d_d;
            CHECK( cudaMalloc((void**) &d_d, (N-k)*sizeof(double)) );
            product_gpu<<<grid, block>>>(d_b, d_ct, d_d, N-k, 1, r-1);
            cudaDeviceSynchronize();
            //cudaMemcpy(d, d_d, (N-k)*sizeof(double), cudaMemcpyDeviceToHost);

            subtract_gpu<<<grid, block>>>(d_a, d_d, N-k, 1);
            cudaDeviceSynchronize();
            //cudaMemcpy(a, d_a, (N-k)*sizeof(double), cudaMemcpyDeviceToHost);

            /*cout << "matrix B: " << endl;
                display(b, N-k, r-1);
                cout << endl;
                cout << "matrix C: " << endl;
                display(c, 1, r-1);
                cout << endl;
                cout << "matrix Ct: " << endl;
                display(ct, r-1, 1);
                cout << endl;
                cout << "matrix D: " << endl;
                display(d, N-k, 1);
                cout << endl;
                cout << "A meno D: " << endl;
                display(a, N-k, 1);
                cout << endl;              */

            //free(b);
            cudaFree(d_b);
            //free(c);
            cudaFree(d_c);
            //free(ct);
            cudaFree(d_ct);
            //free(d);
            cudaFree(d_d);
        }

        cp_array_to_matrix<<<grid, block>>>(L, d_a, N, r, k);
        cudaDeviceSynchronize();

        cudaMemcpy(h_L, L, N*N*sizeof(double), cudaMemcpyDeviceToHost);
        
        //free(a);
        cudaFree(d_a);

        if(h_L[k*N + r-1] > tol) {
            h_L[k*N + r-1] = sqrt(h_L[k*N + r-1]);

            if (k+1 < N)
                for(int i = k+1; i < N; i++)
                    h_L[i*N + r-1] = h_L[i*N + r-1] / h_L[k*N + r-1]; 
        }
        else
            r = r-1;

        cudaMemcpy(L, h_L, N*N*sizeof(double), cudaMemcpyHostToDevice);
    }

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

    cudaFree(G);
    cudaFree(Gt);

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
    cudaDeviceSynchronize();
    cudaMemcpy(A, d_A, M*M*sizeof(double), cudaMemcpyDeviceToHost);
    
    cout << "\n----- A -----\n";
    display<double>(A, M, M);

    cout << "\n----- S -----\n";
    int rank = full_rank_cholesky_decomposition_gpu(d_A, d_S, M, block, grid);
    cudaMemcpy(S, d_S, M*M*sizeof(double), cudaMemcpyDeviceToHost);
    display<double>(S, M, M);

    cout << "\n----- L (" << M - rank << " columns dropped)-----\n";
    submatrix_gpu<<<grid, block>>>(d_S, d_L, M, M, 0, M, 0, rank-1);
    cudaMemcpy(L, d_L, M*M*sizeof(double), cudaMemcpyDeviceToHost);
    //display<double>(L, M, rank);

    cout << "\n----- Lt -----\n";
    transpose_gpu<<<grid, block>>>(d_Lt, d_L, M, rank);
    cudaMemcpy(Lt, d_Lt, rank*M*sizeof(double), cudaMemcpyDeviceToHost);
    //display<double>(Lt, rank, M);

    cout << "\n----- Lt * L -----\n";
    product_gpu<<<grid, block>>>(d_Lt, d_L, d_Lt_L, rank, rank, M);
    cudaMemcpy(Lt_L, d_Lt_L, M*M*sizeof(double), cudaMemcpyDeviceToHost);
    //display<double>(Lt_L, rank, rank);

    cout << "\n----- I -----\n";
    set_diagonal<<<grid, block>>>(d_I, rank);

    for (int i = 0; i < rank; i++){
		nodiag_normalize <<<grid, block>>>(d_Lt_L, d_I, rank, i);
		diag_normalize <<<grid, block>>>(d_Lt_L, d_I, rank, i);
		gaussjordan <<<grid, block>>>(d_Lt_L, d_I, rank, i);
		set_zero <<<grid, block>>>(d_Lt_L, d_I, rank, i);
    }
    
    cudaMemcpy(I, d_I, rank*rank*sizeof(double), cudaMemcpyDeviceToHost);
    //display<double>(I, rank, rank);

    double stop = seconds();

    free(Gt);
    free(A);
    free(I);
    free(S);
    free(L);
    free(Lt);
    free(Lt_L);

    cudaFree(d_Gt);
    cudaFree(d_A);
    cudaFree(d_I);
    cudaFree(d_S);
    cudaFree(L);
    cudaFree(Lt);
    cudaFree(Lt_L);
    cudaDeviceReset();


    cout << "\nMoore-Penrose pseudoinverse calculation time on GPU: " << stop - start << " seconds" << endl;
}