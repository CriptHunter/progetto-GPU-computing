#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cmath>
#include <cstring>
using namespace std;
 
// C++ program to calculate Moore-Penrose inverse matrix

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
    memset(lower, 0, sizeof(lower));
  
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

//return the number of column to drop at the end of the lower triangular matrix
int drop_column_count(float* a, int n) {
    int drop = 0;
    for(int j = n-1; j >= 0; j--) {
        for(int i = n-1; i >= 0; i--) {
            //cout << "j = " << j << " i = " << i << " E = " << a[i*n + j] << " drop " << drop << endl;
            if(a[i*n + j] != 0)
                return drop;
        }
        drop++;
    }
    
    return drop;
}

int drop_zero_column(float* a, float* b, int n) {
    int drop = drop_column_count(a, n);
    int k = 0;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n-drop; j++) {
            b[k] = a[i*n + j];
            k++;
        }
    return drop;
}

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

// Driver program
int main()
{
    int N = 6; // number of rows
    int M = 4; // number of columns

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
        G[i] = rand() % 20;

    // G[0] = 1;
    // G[1] = 2;
    // G[2] = 3;
    // G[3] = 4;
    // G[4] = 5;
    // G[5] = 6;
    // G[6] = 7;
    // G[7] = 8;
    // G[8] = 9;
    
    cout << "\n----- G -----\n";
    display<float>(G, N, M);

    cout << "\n----- Gt -----\n";
    transpose(G, Gt, N, M);
    display<float>(Gt, M, N);

    cout << "\n----- A -----\n";
    multiply(Gt, M, N, G, N, M, A); 
    display(A, M, M);

    cout << "\n----- S -----\n";
    cholesky_decomposition(A, S, M);
    display(S, M, M);

    int drop = drop_zero_column(S, L, M);

    cout << "\n----- L("<< drop << " columns dropped) -----\n";
    display(L, M, M-drop);

    cout <<"\n----- Lt -----\n";
    transpose(L, Lt, M, M-drop);
    display(Lt, M-drop, M);

    cout << "\n----- Lt * L ----- \n";
    multiply(Lt, M-drop, M, L, M, M-drop, Lt_L);
    display(Lt_L, M-drop, M-drop);

    cout << "\n----- I -----\n";
    inverse(Lt_L, I, M-drop);
    display(I, M-drop, M-drop);

    cout << "\n----- Y -----\n";
    float* tmp = new float[M*M];
    float* tmp1 = new float[M*M];
    float* tmp2 = new float[M*M];
    multiply(L, M, M-drop, I, M-drop, M-drop, tmp);
    multiply(tmp, M, M-drop, I, M-drop, M-drop, tmp1);
    multiply(tmp1, M, M-drop, Lt, M-drop, M, tmp2); 
    multiply(tmp2, M, M, Gt, M, N, Y);
    display(Y, M, N);

    free(G);
    free(Gt);
    free(A);
    free(Y);
    free(I);
    free(S);
    free(Lt_L);
    free(tmp);
    free(tmp1);
    free(tmp2);

    return 0;
}
