// C++ program to calculate Moore-Penrose inverse matrix
#include "common.h"

template<class T>
void display(T *a, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            cout << a[i * col + j] << "\t";
        cout << endl;
    }
}

template<class T> 
void display_1D(T* a) {
    for(int i = 0; i < 20; i++) 
        cout << a[i] << " ";
    cout << endl;
}

//transpose a N*M matrix
void transpose(double *a, double *a_t, int row, int col) {
    int k = 0;
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
             a_t[k++] = a[j * col + i];
        }
    }
}
 
//product of two matrix
void multiply(double *a, int row1, int col1, double *b, int row2, int col2, double* matrix_mult) {
    if(col1 != row2)
        cout << "dimensione colonna 1 diversa da dimensione riga 2!\n";

    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            double sum = 0;
            for (int k = 0; k < col1; k++)
                sum = sum + a[i * col1 + k] * b[k * col2 + j];
            matrix_mult[i * col2 + j] = sum;
        }
    }
}

void subtract(double* a, double* b, int row, int col) {
    for(int i = 0; i < row*col; i++) {

        if(a[i] - b[i] < 1E-7 && a[i] - b[i] >= 0)
            a[i] = 0;
        else
            a[i] = a[i] - b[i];

    }
}
 
// Initializing identity matrix with 1 on diagonal
void init_identity(double* a, int n) {
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            if(j == (i))
                a[i*n + j] = 1;
            else
                a[i*n + j] = 0;
            
        }
    }
}

//partial pivoting using element one of each row
void pivot_matrix(double* matrix, double* inverse, int n) {
    double d = 0.0;
    for(int i = n; i > 1; i--) {
        if(matrix[(i-1)*n + 1] < matrix[i*n + 1])
            for(int j = 0; j < n; j++) {
                d = matrix[i*n +j];
                matrix[i*n + j] = matrix[(i-1)*n + j];
                matrix[(i-1)*n + j] = d;

                d = inverse[i*n + j];
                inverse[i*n + j] = inverse[(i-1)*n + j];
                inverse[(i-1)*n + j] = d;
            }
    }
}

//reduce the matrix to a diagonal one (values outside diagonal are 0)
void diagonal_reduce(double* matrix, double* inverse, int n) {
    double d = 0.0;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(j != i) {
                d = matrix[j*n + i] / matrix[i*n + i];   
                    for (int k = 0; k < n; k++) {
                        matrix[j*n + k] -= matrix[i*n + k] * d;
                        inverse[j*n + k] -= inverse[i*n + k] * d;
                    }
            }
        }
    }
}

//reduce the matrix to identity obtaining the inverse in the second matrix
void identity_reduce(double* matrix, double* inverse, int n) {
    double d = 0.0;
    for(int i = 0; i < n; i++) {
        d = matrix[i*n + i];
            for(int j = 0; j < n; j++) {
                matrix[i*n +j] = matrix[i*n + j]/d;
                inverse[i*n +j] = inverse[i*n + j]/d;
            }
    }
}

void inverse(double* matrix, double* inverse, int n) {
    init_identity(inverse, n);
    pivot_matrix(matrix, inverse, n);
    diagonal_reduce(matrix, inverse, n);
    identity_reduce(matrix, inverse, n);
}

//return a submatrix given rows and columns indices
double* submatrix(double* A, int n, int m, int row_start, int row_end, int col_start, int col_end) {
    int n_rows = row_end - row_start + 1;
    int n_cols = col_end - col_start + 1;
    double* sub = (double*) malloc(n_rows*n_cols*sizeof(double));

    int k = 0;
    for(int i = row_start; i <= row_end; i++)
        for(int j = col_start; j <= col_end; j++) {
            sub[k] = A[i*m + j];
            k++;
        }
    return sub;
}

int full_rank_cholesky_decomposition(double* A, double* L, int n) {
    double tol = A[0];
    for(int i = 0; i < n; i++) {
        double k = A[i*n + i];
        if(k < tol)
            tol = k;
    }
    tol = tol * 1E-9;

    int r = 0;

    for(int k = 0; k < n; k++) {
        r = r+1;

        double* a = submatrix(A, n, n, k, n-1, k, k);

        if(r-2 >= 0) {
            double* b = submatrix(L, n, n, k, n-1, 0, r-2);
            double* c = submatrix(L, n, n, k, k, 0, r-2);
            double* ct = (double*) malloc((r-1)*sizeof(double));
            transpose(c, ct, 1, r-1);
            double* d = (double*) malloc((n-k)*sizeof(double));
            multiply(b, n-k, r-1, ct, r-1, 1, d);
            subtract(a, d, n-k, 1);

            /*cout << "matrix A: " << endl;
            display(a, n-k, r-1);
            cout << endl;
            cout << "matrix B: " << endl;
            display(b, n-k, r-1);
            cout << endl;
            cout << "matrix C " << endl;
            display(c, 1, r-1);
            cout << endl;
            cout << "matrix Ct: " << endl;
            display(ct, r-1, 1);
            cout << endl;
            cout << "matrix D: " << endl;
            display(d, n-k, 1);
            cout << endl;*/

            free(b);
            free(c);
            free(ct);
            free(d);
        }

        for(int i = k; i < n; i++)
            L[i*n + r-1] = a[i-k];
        
        free(a);

        if(L[k*n + r-1] > tol) {
            L[k*n + r-1] = sqrt(L[k*n + r-1]);

            if (k+1 < n)
                for(int i = k+1; i < n; i++)
                    L[i*n + r-1] = L[i*n + r-1] / L[k*n + r-1]; 
        }
        else
            r = r-1;
    }

    return r;
}

void drop_zero_column(double* a, double* b, int n, int rank) {
    int k = 0;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < rank; j++) {
            b[k] = a[i*n + j];
            k++;
        }    
}

void geninv(double* G, double* Y, int N, int M)
{
    int old_M = M;
    bool transposed = false;
    double* Gt = (double *) malloc(M*N*sizeof(double)); // transpose of G
    double* A; // Gt * G
    double* S; // lower triangular of A
    double* L; // lower triangular with zero columns dropped
    double* Lt; // upper triangular with zero rows dropped
    double* Lt_L; // Lt * L
    double* I; // inverse of Lt * L

    double start = seconds();
    
    transpose(G, Gt, N, M); // transpose G in Gt

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
        multiply(G, N, old_M, Gt, old_M, N, A); // A = G * Gt 
    else
        multiply(Gt, old_M, N, G, N, old_M, A); // A = Gt * G

    int rank = full_rank_cholesky_decomposition(A, S, M); // S = cholesky(A)

    drop_zero_column(S, L, M, rank); // L = S without zero columns
    transpose(L, Lt, M, rank);
    multiply(Lt, rank, M, L, M, rank, Lt_L); // Lt_L = Lt * L

    inverse(Lt_L, I, rank); // I = inverse(Lt_L)

    double* tmp;
    double* tmp1; 
    double* tmp2;

    if(transposed) { // Y = Gt * L * I * I * Lt
        tmp =  (double *) malloc(old_M*rank*sizeof(double));
        tmp1 = (double *) malloc(old_M*rank*sizeof(double));
        tmp2 = (double *) malloc(old_M*rank*sizeof(double));

        multiply(Gt, old_M, N, L, N, rank, tmp);
        multiply(tmp, old_M, rank, I, rank, rank, tmp1);
        multiply(tmp1, old_M, rank, I, rank, rank, tmp2);
        multiply(tmp2, old_M, rank, Lt, rank, N, Y);
    }
    else { // Y = L * I * I * Lt * Gt
        tmp =  (double *) malloc(old_M*rank*sizeof(double));
        tmp1 = (double *) malloc(old_M*rank*sizeof(double));
        tmp2 = (double *) malloc(old_M*old_M*sizeof(double));

        multiply(L, M, rank, I, rank, rank, tmp);
        multiply(tmp, M, rank, I, rank, rank, tmp1);
        multiply(tmp1, M, rank, Lt, rank, M, tmp2); 
        multiply(tmp2, M, M, Gt, M, N, Y);
    }

    double stop = seconds();
    cout << "\nMoore-Penrose pseudoinverse calculation time on CPU: " << stop - start << " seconds" << endl;

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
}

void geninv_verbose(double* G, double* Y, int N, int M)
{
    int old_M = M;
    bool transposed = false;
    double* Gt = (double *) malloc(M*N*sizeof(double)); // transpose of G
    double* A; // Gt * G
    double* S; // lower triangular of A
    double* L; // lower triangular with zero columns dropped
    double* Lt; // upper triangular with zero rows dropped
    double* Lt_L; // Lt * L
    double* I; // inverse of Lt * L

    time_t start, end;
    time(&start); 
    
    cout << "\n----- G -----\n";
    display<double>(G, N, M);

    cout << "\n----- Gt -----\n";
    transpose(G, Gt, N, M); // transpose G in Gt
    display<double>(Gt, M, N);

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

    cout << "\n----- A -----\n";
    if(transposed)
        multiply(G, N, old_M, Gt, old_M, N, A); // A = G * Gt 
    else
        multiply(Gt, old_M, N, G, N, old_M, A); // A = Gt * G

    display(A, M, M);

    cout << "\n----- S -----\n";
    int rank = full_rank_cholesky_decomposition(A, S, M); // S = cholesky(A)
    display(S, M, M);
    
    cout << "\n----- L("<< M-rank << " columns dropped) -----\n";    
    drop_zero_column(S, L, M, rank); // L = S without zero columns
    display(L, M, rank);

    cout <<"\n----- Lt -----\n";
    transpose(L, Lt, M, rank);
    display(Lt, rank, M);

    cout << "\n----- Lt * L ----- \n";
    multiply(Lt, rank, M, L, M, rank, Lt_L); // Lt_L = Lt * L
    display(Lt_L, rank, rank);

    cout << "\n----- I -----\n";
    inverse(Lt_L, I, rank); // I = inverse(Lt_L)
    display(I, rank, rank);

    cout << "\n----- Y -----\n";
    double* tmp;
    double* tmp1; 
    double* tmp2;

    if(transposed) { // Y = Gt * L * I * I * Lt
        tmp =  (double *) malloc(old_M*rank*sizeof(double));
        tmp1 = (double *) malloc(old_M*rank*sizeof(double));
        tmp2 = (double *) malloc(old_M*rank*sizeof(double));

        multiply(Gt, old_M, N, L, N, rank, tmp);
        multiply(tmp, old_M, rank, I, rank, rank, tmp1);
        multiply(tmp1, old_M, rank, I, rank, rank, tmp2);
        multiply(tmp2, old_M, rank, Lt, rank, N, Y);
    }
    else { // Y = L * I * I * Lt * Gt
        tmp =  (double *) malloc(old_M*rank*sizeof(double));
        tmp1 = (double *) malloc(old_M*rank*sizeof(double));
        tmp2 = (double *) malloc(old_M*old_M*sizeof(double));

        multiply(L, M, rank, I, rank, rank, tmp);
        multiply(tmp, M, rank, I, rank, rank, tmp1);
        multiply(tmp1, M, rank, Lt, rank, M, tmp2); 
        multiply(tmp2, M, M, Gt, M, N, Y);
    }

    display(Y, old_M, N);

    time(&end);
    cout << "\nMoore-Penrose pseudoinverse calculation time on CPU: " << (double)(end-start) << " seconds" << endl;

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
}


