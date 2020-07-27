#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cmath>
#include <cstring>
using namespace std;
 
// C++ program to calculate Moore-Penrose inverse matrix

#define N 5 //number of rows
#define M 4 //number of columns

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

// decompose a simmetric matrix into a lower triangular and his transpose
// this function return only the lower triangular
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

template<class T>
void display(T *a, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            cout << a[i * col + j] << " ";
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
    float* matrix = new float[N*M]; // A matrix
    float* t_matrix = new float[M*N]; // A_t matrix
    float* matrix_mult = new float[M*M]; // A_t * A matrix
    float* pseudoinverse = new float[M*N]; // A+ matrix
    float* inv = new float[M*M]; // (A_t * A)^-1 matrix
    float* l_tri_low = new float[M*M]; // lower triangular of (A_t * A)
    float* l_tri_up = new float[M*M];  // upper triangular of (A_t * A)
    float* l_tri_up_low = new float[M*M]; // upper tri * lower tri

    srand(time(NULL)); 

    for(int i = 0; i < N * M; i ++)
        matrix[i] = rand() % 20;
    
    cout << "\nA:\n";
    display<float>(matrix, N, M);

    cout << "\nA_t:\n";
    transpose(matrix, t_matrix, N, M);
    display<float>(t_matrix, M, N);

    cout << "\nA_t * A:\n";
    multiply(t_matrix, M, N, matrix, N, M, matrix_mult); 
    display(matrix_mult, M, M);

    cout << "\nLower triangular of A_t * A\n";
    cholesky_decomposition(matrix_mult, l_tri_low, M);
    display(l_tri_low, M, M);

    cout <<"\nUpper triangular of A_t * A\n";
    transpose(l_tri_low, l_tri_up, M, M);
    display(l_tri_up, M, M);

    cout << "\nUpper tri * Lower tri\n";
    multiply(l_tri_up, M, M, l_tri_low, M, M, l_tri_up_low);
    display(l_tri_up_low, M, M);

    cout << "\n(Upper tri * Lower tri)^-1:\n";
    inverse(l_tri_up_low, inv, M);
    display(inv, M, M);

    cout << "\nThe Moore-Penrose inverse is :\n";
    float* matrix_mult_1 = new float[M*M];
    float* matrix_mult_2 = new float[M*M];
    multiply(l_tri_low, M, M, inv, M, M, matrix_mult);
    multiply(matrix_mult, M, M, inv, M, M, matrix_mult_1);
    multiply(matrix_mult_1, M, M, l_tri_up, M, M, matrix_mult_2);
    multiply(matrix_mult_2, M, M, t_matrix, M, N, pseudoinverse);
    display(pseudoinverse, M, N);

    // multiply(inv, M, M, t_matrix, M, N, pseudoinverse);
    // cout << "\nThe Monroe-penrose inverse is :\n";
    // display(pseudoinverse, M, N);

    free(matrix);
    free(t_matrix);
    free(matrix_mult);
    free(pseudoinverse);
    free(inv);
    free(l_tri_low);
    free(l_tri_up);
    free(matrix_mult_1);
    free(matrix_mult_2);

    return 0;
}
