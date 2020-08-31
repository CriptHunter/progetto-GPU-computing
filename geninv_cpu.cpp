/**
 * Print a matrix on std output
 * @param A Input matrix
 * @param N number of rows
 * @param M number of columns
 */
template<class T>
void display(T *A, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++)
            cout << A[i * M + j] << "\t";
        cout << endl;
    }
}

/**
 * Matrix transpose
 * @param A Input matrix
 * @param At Transpose of `A`
 * @param N number of rows
 * @param M number of columns
 */
void transpose(double *A, double *At, int N, int M) {
    int k = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
             At[k++] = A[j * M + i];
        }
    }
}

/**
 * Matrix product
 * @param A First matrix
 * @param N1 number of rows of `A`
 * @param M1 number of columns of `A`
 * @param B Second matrix
 * @param N2 number of rows of `B`
 * @param M2 number of columns of `B`
 * @param C `A` * `B`
 */
void product(double *A, int N1, int M1, double *B, int N2, int M2, double* C) {
    if(M1 != N2)
        cout << "dimensione colonna 1 diversa da dimensione riga 2!\n";

    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < M2; j++) {
            double sum = 0;
            for (int k = 0; k < M1; k++)
                sum = sum + A[i * M1 + k] * B[k * M2 + j];
            C[i * M2 + j] = sum;
        }
    }
}

/**
 * Matrix subtraction
 * @param A first matrix and result
 * @param B second matrix
 * @param N number of rows of `A` and `B`
 * @param M number of columns of `A` and `B`
 */
void subtract(double* A, double* B, int N, int M) {
    for(int i = 0; i < N*M; i++)
        A[i] = A[i] - B[i];
}

/**
 * initialize identity matrix
 * @param A input matrix
 * @param N matrix order
 */
void init_identity(double* A, int N) {
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            if(j == (i))
                A[i*N + j] = 1;
            else
                A[i*N + j] = 0;
            
        }
    }
}

/**
 * Matrix inverse using Gauss-Jordan method
 * @param A Input square matrix
 * @param I Inverse of `A`
 * @param N matrix order
 */
void inverse(double* A, double* I, int N) {
    init_identity(I, N);
    double A1;
    double A2;
    double A3;
    double I1;
    double I2;

    // c_row is the current row
    for(int c_row = 0; c_row < N; c_row++) {

        // divide element outside diagonal
        for(int col = 0; col < N; col++) {
            if(col != c_row) {
                I[c_row*N + col] /= A[c_row*N + c_row];
                A[c_row*N + col] /= A[c_row*N + c_row];
            }
        }

        // divide diagonal element
        I[c_row*N + c_row] = I[c_row*N + c_row] / A[c_row*N + c_row];
        A[c_row*N + c_row]  = 0;

        // gauss jordan
        for(int row = 0; row < N; row++) {
            for(int col = 0; col < N; col++) {
                if (row != c_row) {

                    A2 = A[row*N + c_row];
                    I1 = I[row*N + col];
                    I2 = I[c_row*N + col];

                    I1 -= I2 * A2;
                    I[row*N + col] = I1;
                    
                    if (col != c_row) {
                        A1 = A[row*N + col];
                        A3 = A[c_row*N + col];
                        A1 -= A3 * A2;
                        A[row*N + col] = A1;
                    }
                }
            }
        }
    }
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
void submatrix(double* B, double* A, int N, int M, int row_start, int row_end, int col_start, int col_end) {
    int k = 0;
    for(int i = row_start; i <= row_end; i++)
        for(int j = col_start; j <= col_end; j++) {
            B[k] = A[i*M + j];
            k++;
        }
}

/**
 * Subtract two submatrices
 * @param C Result of `A` - `B`
 * @param A first matrix
 * @param B second matrix
 * @param M number of columns of `A`
 * @param M2 number of columns of `B`
 */
void submatrix_subtract(double* C, double* A, double* B, int M, int M2, int row_start, int row_end, int col_start, int col_end, 
                                                                        int row_start2, int row_end2, int col_start2, int col_end2)
{
    int n_rows1 = row_end - row_start;
    int n_cols1 = col_end - col_start;
    int n_rows2 = row_end2 - row_start2;
    int n_cols2 = col_end2 - col_start2;

    if(n_rows1 != n_rows2 || n_cols1 != n_cols2) {
        cout << "submatrix_subtract = numero di righe / colonne diverso" << endl;
        return;
    }
    int k = 0;

    for(int i = 0; i <= n_rows1; i++)
        for(int j = 0; j <= n_cols1; j++) {
            C[k] = A[(i+row_start)*M + j + col_start] - B[(i+row_start2)*M2 + j + col_start2];
            k++;
        }
}        
                                  
/**
 * Cholesky decomposition for rank deficient matrix
 * @param A input square matrix
 * @param L lower triangular matrix of `A`
 * @param N number of row/col of `A`
 * @return rank of `L`
*/
int full_rank_cholesky_decomposition(double* A, double* L, int N) {
    memset(L, 0, N*N*sizeof(double));
    double* a = (double*) malloc(N*sizeof(double));
    double* b = (double*) malloc(N*N*sizeof(double));
    double* c = (double*) malloc(N*sizeof(double));
    double* d = (double*) malloc(N*sizeof(double));

    //tollerance threshold for sqrt, it's the diagonal minimum
    double tol = A[0];
    for(int i = 0; i < N; i++) {
        double k = A[i*N + i];
        if(k < tol)
            tol = k;
    }
    tol = tol * 1E-9;

    int r = 0;

    for(int k = 0; k < N; k++) {
        r = r+1;

        if(r < 2)
            submatrix(a, A, N, N, k, N-1, k, k); // A(k:N-1, k:k)
        else {
            submatrix(b, L, N, N, k, N-1, 0, r-2); // L(k:N-1, 0:r-2)
            submatrix(c, L, N, N, k, k, 0, r-2);  // transpose of L(k:k, 0:r-2)
            product(b, N-k, r-1, c, r-1, 1, d);  
            submatrix_subtract(a, A, d, N, 1, k, N-1, k, k, 0, N-k-1, 0, 0);
        }
        //copying result non zero values, each time it copies one element less to form a triangular matrix
        for(int i = k; i < N; i++)
            L[i*N + r-1] = a[i-k];
        
        if(L[k*N + r-1] > tol) {
            L[k*N + r-1] = sqrt(L[k*N + r-1]);

            if (k+1 < N)
                for(int i = k+1; i < N; i++)
                    L[i*N + r-1] = L[i*N + r-1] / L[k*N + r-1]; 
        }
        else
            r = r-1;
    }

    free(a);
    free(b);
    free(c);
    free(d);

    return r;
}

/**
 * Drop columns composed by zeroes of a matrix
 * @param A input square matrix
 * @param B `A` with dropped columns
 * @param N number of rows/columns of `A`
 * @return rank of `A`
*/
void drop_zero_column(double* A, double* B, int N, int rank) {
    int k = 0;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < rank; j++) {
            B[k] = A[i*N + j];
            k++;
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
double geninv(double* G, double* Y, int N, int M) {
    int old_M = M; // to remember M original value
    bool transposed = false; // true if N < M
    double* Gt = (double *) malloc(M*N*sizeof(double)); // transpose of G
    double* A; // Gt * G
    double* S; // lower triangular of A
    double* L; // lower triangular with zero columns dropped
    double* Lt; // upper triangular with zero rows dropped
    double* Lt_L; // Lt * L
    double* I; // inverse of Lt * L

    double start = seconds();
    
    transpose(G, Gt, N, M); // transpose G in Gt

    //pseudoinverse formula is different if N < M
    if(N < M) {
        transposed = true;
        M = N;
    }

    A    = (double *) malloc(M*M*sizeof(double)); // Gt * G
    S    = (double *) malloc(M*M*sizeof(double)); // lower triangular of A
    L    = (double *) malloc(M*M*sizeof(double)); // lower triangular with zero columns dropped
    Lt   = (double *) malloc(M*M*sizeof(double)); // upper triangular with zero rows dropped
    Lt_L = (double *) malloc(M*M*sizeof(double)); // Lt * L
    I    = (double *) malloc(M*M*sizeof(double)); // inverse of Lt * L

    if(transposed)
        product(G, N, old_M, Gt, old_M, N, A); // A = G * Gt 
    else
        product(Gt, old_M, N, G, N, old_M, A); // A = Gt * G

    int rank = full_rank_cholesky_decomposition(A, S, M); // S = cholesky(A)
    drop_zero_column(S, L, M, rank); // L = S without zero columns
    transpose(L, Lt, M, rank);
    product(Lt, rank, M, L, M, rank, Lt_L); // Lt_L = Lt * L
    inverse(Lt_L, I, rank); // I = inverse(Lt_L)

    double* tmp;
    double* tmp1; 
    double* tmp2;

    if(transposed) { // Y = Gt * L * I * I * Lt
        tmp =  (double *) malloc(old_M*rank*sizeof(double));
        tmp1 = (double *) malloc(old_M*rank*sizeof(double));
        tmp2 = (double *) malloc(old_M*rank*sizeof(double));

        product(Gt, old_M, N, L, N, rank, tmp);
        product(tmp, old_M, rank, I, rank, rank, tmp1);
        product(tmp1, old_M, rank, I, rank, rank, tmp2);
        product(tmp2, old_M, rank, Lt, rank, N, Y);
    }
    else { // Y = L * I * I * Lt * Gt
        tmp =  (double *) malloc(old_M*rank*sizeof(double));
        tmp1 = (double *) malloc(old_M*rank*sizeof(double));
        tmp2 = (double *) malloc(old_M*old_M*sizeof(double));

        product(L, M, rank, I, rank, rank, tmp);
        product(tmp, M, rank, I, rank, rank, tmp1);
        product(tmp1, M, rank, Lt, rank, M, tmp2); 
        product(tmp2, M, M, Gt, M, N, Y);
    }

    double stop = seconds();

    free(Gt);
    free(A);
    free(I);
    free(S);
    free(L);
    free(Lt);
    free(Lt_L);
    free(tmp);
    free(tmp1);
    free(tmp2);

    return stop - start;
}

/**
 * find least squares of a linear system using moore-penrose pseudoinverse
 * @param A Pseudoinverse
 * @param x variables vector
 * @param y constants vector
 * @param N number of equations 
 * @param M number of variables
 */
void least_square(double* A, double* x, double* y, int N, int M) {
    product(A, M, N, y, N, 1, x);
}




