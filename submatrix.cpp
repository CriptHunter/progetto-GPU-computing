#include <bits/stdc++.h> 
using namespace std; 
    
template<class T>
void display(T *a, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            cout << a[i * col + j] << "\t";
        cout << endl;
    }
}

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

    display(sub, n_rows, n_cols);
    return sub;
}
 
int main() 
{ 
    int N = 6;
    int M = 4;
    float* G = new float[N*M];

    srand(time(NULL));
    for(int i = 0; i < N * M; i ++)
        G[i] = i+1;

    display(G, N, M);

    cout << endl;

    submatrix(G, N, M, 1, 5, 1, 3);

    return 0; 
} 