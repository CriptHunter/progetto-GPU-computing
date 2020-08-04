#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cmath>
#include <cstring>
using namespace std;
 
// C++ program to calculate Moore-Penrose inverse matrix

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
void transpose(float *a, float *a_t, int row, int col) {
    int k = 0;
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
             a_t[k++] = a[j * col + i];
        }
    }
}
 
//multiply two matrix
void multiply(float *a, int row1, int col1, float *b, int row2, int col2, float* matrix_mult) {
    if(col1 != row2)
        cout << "dimensione colonna 1 diversa da dimensione riga 2!\n";

    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            float sum = 0;
            for (int k = 0; k < col1; k++)
                sum = sum + a[i * col1 + k] * b[k * col2 + j];
            matrix_mult[i * col2 + j] = sum;
        }
    }
}

void subtract(float* a, float* b, int row, int col) {
    for(int i = 0; i < row*col; i++) {

        if(a[i] - b[i] < 1E-7)
            a[i] = 0;
        else
            a[i] = a[i] - b[i];
        
    }
}
 
// Initializing identity matrix with 1 on diagonal
void init_identity(float* a, int n) {
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
void pivot_matrix(float* matrix, float* inverse, int n) {
    float d = 0.0;
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
void diagonal_reduce(float* matrix, float* inverse, int n) {
    float d = 0.0;
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
void identity_reduce(float* matrix, float* inverse, int n) {
    float d = 0.0;
    for(int i = 0; i < n; i++) {
        d = matrix[i*n + i];
            for(int j = 0; j < n; j++) {
                matrix[i*n +j] = matrix[i*n + j]/d;
                inverse[i*n +j] = inverse[i*n + j]/d;
            }
    }
}

void inverse(float* matrix, float* inverse, int n) {
    init_identity(inverse, n);
    pivot_matrix(matrix, inverse, n);
    diagonal_reduce(matrix, inverse, n);
    identity_reduce(matrix, inverse, n);
}

// return the lower triangular matrix of a symmetric matrix
void cholesky_decomposition(float* matrix, float* lower, int n) { 
    memset(lower, 0, sizeof(n));
  
    // Decomposing a matrix into Lower Triangular 
    for (int i = 0; i < n; i++) { 
        for (int j = 0; j <= i; j++) { 
            float sum = 0.0; 
  
            if (j == i) { 
                for (int k = 0; k < j; k++) 
                    sum += pow(lower[j*n + k], 2); 
                lower[j*n +j] = sqrt(matrix[j*n + j] - sum); 
            } 
        
            else { 
                for (int k = 0; k < j; k++) 
                    sum += (lower[i*n + k] * lower[j*n + k]); 
                    lower[i*n + j] = (matrix[i*n + j] - sum) / lower[j*n + j]; 
            } 
        } 
    } 
}

//return a submatrix given rows and columns indices
float* submatrix(float* A, int n, int m, int row_start, int row_end, int col_start, int col_end) {
    int n_rows = row_end - row_start + 1;
    int n_cols = col_end - col_start + 1;
    float* sub = new float[n_rows*n_cols];

    int k = 0;
    for(int i = row_start; i <= row_end; i++)
        for(int j = col_start; j <= col_end; j++) {
            sub[k] = A[i*m + j];
            k++;
        }
    return sub;
}

int full_rank_cholesky_decomposition(float* A, float* L, int n) {
    float tol = 1E-9;
    int r = 0;

    for(int k = 0; k < n; k++) {
        r = r+1;
      
        float* a = submatrix(A, n, n, k, n-1, k, k);

        if(r-2 >= 0) {
            float* b = submatrix(L, n, n, k, n-1, 0, r-2);
            float* c = submatrix(L, n, n, k, k, 0, r-2);
            float* ct = new float[r-1];
            transpose(c, ct, 1, r-1);
            float* d = new float[(n-k)*1];
            multiply(b, n-k, r-1, ct, r-1, 1, d);
            subtract(a, d, n-k, 1);
        }

        for(int i = k; i < n; i++)
            L[i*n + r-1] = a[i-k];

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

void drop_zero_column(float* a, float* b, int n, int rank) {
    int k = 0;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < rank; j++) {
            b[k] = a[i*n + j];
            k++;
        }    
}

// Driver program
int main()
{
    int N = 6; // number of rows
    int M = 3; // number of columns

    float* G = new float[N*M]; // start matrix
    float* Gt = new float[M*N]; // transpose of G
    float* A = new float[M*M]; // Gt * G
    float* S = new float[M*M]; // lower triangular of A
    float* L = new float[M*M]; // lower triangular with zero columns dropped
    float* Lt = new float[M*M]; // upper triangular with zero rows dropped
    float* Lt_L = new float[M*M]; // Lt * L
    float* I = new float[M*M]; // inverse of Lt * L
    float* Y = new float[M*N]; // pseudoinverse

    srand(time(NULL));

    for(int i = 0; i < N * M; i ++)
        G[i] = i+1;
        //G[i] = rand() % 2;

    
    cout << "\n----- G -----\n";
    display<float>(G, N, M);

    cout << "\n----- Gt -----\n";
    transpose(G, Gt, N, M);
    display<float>(Gt, M, N);

    cout << "\n----- A -----\n";
    multiply(Gt, M, N, G, N, M, A); 
    display(A, M, M);

    cout << "\n----- S -----\n";
    int rank = full_rank_cholesky_decomposition(A, S, M);
    display(S, M, rank);
    
    cout << "\n----- L("<< M-rank << " columns dropped) -----\n";    
    drop_zero_column(S, L, M, rank);
    display(L, M, rank);

    cout <<"\n----- Lt -----\n";
    transpose(L, Lt, M, rank);
    display(Lt, rank, M);

    cout << "\n----- Lt * L ----- \n";
    multiply(Lt, rank, M, L, M, rank, Lt_L);
    display(Lt_L, rank, rank);

    cout << "\n----- I -----\n";
    inverse(Lt_L, I, rank);
    display(I, rank, rank);

    cout << "\n----- Y -----\n";
    float* tmp = new float[M*M];
    float* tmp1 = new float[M*M];
    float* tmp2 = new float[M*M];
    multiply(L, M, rank, I, rank, rank, tmp);
    multiply(tmp, M, rank, I, rank, rank, tmp1);
    multiply(tmp1, M, rank, Lt, rank, M, tmp2); 
    multiply(tmp2, M, M, Gt, M, N, Y);
    display(Y, M, N);

    delete G;
    delete Gt;
    delete A;
    delete Y;
    delete I;
    delete S;
    delete Lt_L;
    delete tmp;
    delete tmp1;
    delete tmp2;

    return 0;
}
