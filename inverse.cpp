/**
 * Inverse of a Matrix
 * Gauss-Jordan Elimination
 **/

#include<iostream>
using namespace std;

void print(float* mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            cout << mat[i*n +j] << " ";
        cout << endl;
    }
    cout << endl;
}

int main()
{
    int i = 0, j = 0, k = 0, n = 4;
    float *mat = new float[n*n];
    float *inv = new float[n*n];
    float d = 0.0;
    
    for(i = 0; i < n*n; i++)
        mat[i] = rand() % 20;

    cout << endl << "Input matrix:" << endl;
    print(mat, n);

    inverse(mat, inv, n);
    
    // // Initializing identity matrix with 1 on diagonal
    // for(i = 0; i < n; ++i) {
    //     for(j = 0; j < n; ++j) {
    //         if(j == (i))
    //             inv[i*n + j] = 1;
    //     }
    // }

    // cout << endl << "Identity matrix:" << endl;
    // print(inv, n);
    
    // //pivoting matrix
    // for(i = n; i > 1; i--) {
    //     if(mat[(i-1)*n + 1] < mat[i*n + 1])
    //         for(j = 0; j < n; j++) {
    //             d = mat[i*n +j];
    //             mat[i*n + j] = mat[(i-1)*n + j];
    //             mat[(i-1)*n + j] = d;

    //             d = inv[i*n + j];
    //             inv[i*n + j] = inv[(i-1)*n + j];
    //             inv[(i-1)*n + j] = d;
    //         }
    // }

    // cout << endl << "Pivoted matrix:" << endl;
    // print(mat, 4);
    // cout << endl << "Pivoted identity matrix:" << endl;
    // print(inv, 4);

    // // calculate diagonal matrix
    // for(i = 0; i < n; i++) {
    //     for(j = 0; j < n; j++) {
    //         if(j != i) {
    //             d = mat[j*n + i] / mat[i*n + i];   
    //             for (k = 0; k < n; k++) {
    //                 mat[j*n + k] -= mat[i*n + k] * d;
    //                 inv[j*n + k] -= inv[i*n + k] * d;
    //             }
    //         }
    //     }
    // }
    
    // cout << endl << "diagonal matrix:" << endl;
    // print(mat, 4);
    // print(inv, 4);

    // //reduce to identity
    // for(i = 0; i < n; i++) {
    //     d = mat[i*n + i];
    //     for(j = 0; j < n; j++) {
    //         mat[i*n +j] = mat[i*n + j]/d;
    //         inv[i*n +j] = inv[i*n + j]/d;
    //     }
    // }

    cout << endl << "inverse matrix:" << endl;
    print(mat, 4);
    print(inv, 4);

    delete(mat);
    delete(inv);
    
    return 0;
}