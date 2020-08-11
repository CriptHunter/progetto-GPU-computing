#include "geninv_cpu.cpp"

int main() {
    int N = 6;
    int M = 4;

    double* G = (double *) malloc(N*M*sizeof(double)); // start matrix
    double* Y = (double *) malloc(M*N*sizeof(double)); // pseudoinverse

    random_matrix(G, N, M);
    printf_matrix(G, N, M, "matrix.txt");
    geninv(G, Y, N, M);
    printf_matrix(Y, M, N, "pseudoinverse.txt");

    free(G);
    free(Y);
}