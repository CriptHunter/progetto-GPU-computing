// Cholesky Decomposition 
#include <bits/stdc++.h> 
using namespace std; 
  
const int MAX = 100; 
  
void print(float* mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            cout << mat[i*n +j] << " ";
        cout << endl;
    }
    cout << endl;
}

void cholesky_Decomposition(float* matrix, float* lower, int n) { 
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
  
// Driver Code 
int main() 
{ 
    int n = 3;
    float* matrix = new float[n*n];
    float* lower = new float[n*n];

    matrix[0] = 4;
    matrix[1] = 12;
    matrix[2] = -16;
    matrix[3] = 12;
    matrix[4] = 37;
    matrix[5] = -43;
    matrix[6] = -16;
    matrix[7] = -43;
    matrix[8] = 98;

 
    cholesky_Decomposition(matrix, lower, n);
    print(lower, n);

    return 0; 
} 


/*
 Lower Triangular                     Transpose
     2       0       0               2       6      -8
     6       1       0               0       1       5
    -8       5       3               0       0       3
*/