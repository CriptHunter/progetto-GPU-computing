// C++ program to calculate Moore-Penrose inverse matrix

template<class T>
void display(T *A, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++)
            cout << A[i * M + j] << "\t";
        cout << endl;
    }
}

//transpose a N*M matrix
void transpose(double *A, double *At, int N, int M) {
    int k = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
             At[k++] = A[j * M + i];
        }
    }
}
 
//product of two matrix
void product(double *A, int row1, int col1, double *B, int row2, int col2, double* C) {
    if(col1 != row2)
        cout << "dimensione colonna 1 diversa da dimensione riga 2!\n";

    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            double sum = 0;
            for (int k = 0; k < col1; k++)
                sum = sum + A[i * col1 + k] * B[k * col2 + j];
            C[i * col2 + j] = sum;
        }
    }
}

void subtract(double* A, double* B, int N, int M) {
    // for(int i = 0; i < N*M; i++) {
    //     if(A[i] - B[i] < 1E-7 && A[i] - B[i] >= 0)
    //         A[i] = 0;
    //     else
    //         A[i] = A[i] - B[i];
    // }
    for(int i = 0; i < N*M; i++)
        A[i] = A[i] - B[i];
}
 
// Initializing identity matrix with 1 on diagonal
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

//partial pivoting using element one of each row
void pivot_matrix(double* A, double* I, int N) {
    double d = 0.0;
    for(int i = N; i > 1; i--) {
        if(A[(i-1)*N + 1] < A[i*N + 1])
            for(int j = 0; j < N; j++) {
                d = A[i*N +j];
                A[i*N + j] = A[(i-1)*N + j];
                A[(i-1)*N + j] = d;

                d = I[i*N + j];
                I[i*N + j] = I[(i-1)*N + j];
                I[(i-1)*N + j] = d;
            }
    }
}

//reduce the matrix to a diagonal one (values outside diagonal are 0)
void diagonal_reduce(double* A, double* I, int N) {
    double d = 0.0;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if(j != i) {
                d = A[j*N + i] / A[i*N + i];   
                    for (int k = 0; k < N; k++) {
                        A[j*N + k] -= A[i*N + k] * d;
                        I[j*N + k] -= I[i*N + k] * d;
                    }
            }
        }
    }
}

//reduce the matrix to identity obtaining the inverse in the second matrix
void identity_reduce(double* A, double* I, int N) {
    double d = 0.0;
    for(int i = 0; i < N; i++) {
        d = A[i*N + i];
            for(int j = 0; j < N; j++) {
                A[i*N +j] = A[i*N + j]/d;
                I[i*N +j] = I[i*N + j]/d;
            }
    }
}

void inverse(double* A, double* I, int N) {
    init_identity(I, N);
    pivot_matrix(A, I, N);
    diagonal_reduce(A, I, N);
    identity_reduce(A, I, N);
}

//return a submatrix given rows and columns indices
double* submatrix(double* A, int N, int M, int row_start, int row_end, int col_start, int col_end) {
    int n_rows = row_end - row_start + 1;
    int n_cols = col_end - col_start + 1;
    double* sub = (double*) malloc(n_rows*n_cols*sizeof(double));

    int k = 0;
    for(int i = row_start; i <= row_end; i++)
        for(int j = col_start; j <= col_end; j++) {
            sub[k] = A[i*M + j];
            k++;
        }
    return sub;
}

int full_rank_cholesky_decomposition(double* A, double* L, int N) {
    memset(L, 0, N*N*sizeof(double));
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

        double* a = submatrix(A, N, N, k, N-1, k, k);

        if(r-2 >= 0) {
            double* b = submatrix(L, N, N, k, N-1, 0, r-2);
            double* c = submatrix(L, N, N, k, k, 0, r-2);
            double* d = (double*) malloc((N-k)*sizeof(double));
            product(b, N-k, r-1, c, r-1, 1, d);
            subtract(a, d, N-k, 1);

            free(b);
            free(c);
            free(d);
        }

        for(int i = k; i < N; i++)
            L[i*N + r-1] = a[i-k];
        
        free(a);

        if(L[k*N + r-1] > tol) {
            L[k*N + r-1] = sqrt(L[k*N + r-1]);

            if (k+1 < N)
                for(int i = k+1; i < N; i++)
                    L[i*N + r-1] = L[i*N + r-1] / L[k*N + r-1]; 
        }
        else
            r = r-1;
    }

    return r;
}

//return a matrix with column from 0 to rank
void drop_zero_column(double* A, double* B, int N, int rank) {
    int k = 0;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < rank; j++) {
            B[k] = A[i*N + j];
            k++;
        }    
}

double geninv(double* G, double* Y, int N, int M)
{
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
    //cout << "\nMoore-Penrose pseudoinverse calculation time on CPU: " << stop - start << " seconds" << endl;

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




