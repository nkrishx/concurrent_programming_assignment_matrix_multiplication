#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <string.h>

#define N 1024 // number of row and col of matrix
#define BLOCK_SIZE 256 // block size for tiling method

int chunk = 0; // number of computations handled by each thread using OpenMP
double percentage = 0.0;

int** allocate_matrix(size_t size);
void free_matrix(int** matrix, size_t size);
int** transpose_matrix(int** A, size_t size);
int** zeros(size_t size);
int** ones(size_t size);

void multiply_matrix_naive(int** A, int** B, int**C, size_t size);
void multiply_matrix_transpose_omp(int** A, int** B, int** C, size_t size);
void multiply_matrix_tiling_2d_with_transposition_omp(int** A, int** B, int** C, size_t matrix_size, size_t block_size );

int** allocate_matrix(size_t size)
{
    int **arr = malloc(size * sizeof(int *));
    for (int i=0; i<size; i++)
        arr[i] = malloc(size * sizeof(int));

    return arr;
}

void free_matrix(int** matrix, size_t size)
{
    for(int i = 0; i< size; i++)
        free(matrix[i]);
    free(matrix);
}

int** transpose_matrix(int** A, size_t size){
    int** TrA = allocate_matrix(size);
    for(int i = 0 ; i < size ; i++){
        for(int j = 0; j < size; j++){
            TrA[i][j] = A[j][i];
        }
    }
    return TrA;
}

// Init a matrix with zeros
int** zeros(size_t size)
{
    int** A = allocate_matrix(size);
    for(int i = 0; i < size; i++)
        for(int j = 0; j < size; j++)
            A[i][j] = 0;
    return A;
}

// fill matrix with half top with ones and the rest with 2
int** ones(size_t size)
{
    int** A = allocate_matrix(size);
    int half_size = size/2;

    for(int i = 0; i < half_size; i++)
        for(int j = 0; j < size; j++)
            A[i][j] = 1;

    for(int i = half_size; i < size; i++)
    {

        for(int j = 0; j < size; j++)
        {
            A[i][j] = 2;
        }

    }

    return A;
}

void multiply_matrix_naive(int** A, int** B, int** C, size_t size)
{
    //int tmp = 0;
    for(int i = 0; i<size; i++){
        for(int j = 0; j< size; j++){
            C[i][j] = 0;

            for(int k = 0; k < size; k++){
                C[i][j] += A[i][k]*B[k][j];
            }

        }
    }
}

void multiply_matrix_transpose_omp(int** A, int** B, int** C, size_t size)
{
    int** TrB = transpose_matrix(B,size);

    int tmp = 0;
    int nthread = omp_get_num_threads();
    chunk = size/nthread;
    int tid ;
        #pragma omp parallel shared(A,B,C,nthread, chunk) private(tid,tmp)
            #pragma omp for schedule(dynamic,chunk)
            for(int i = 0; i < size; i++){
                for(int j = 0; j < size ; j++)
                {
                    tmp = 0;
                    for(int k = 0; k < size; k++)
                        tmp += A[i][k]*TrB[j][k];
                    C[i][j] = tmp;
                }
            }
}

void multiply_matrix_tiling_2d_with_transposition_omp(int** A, int** B, int** C, size_t matrix_size, size_t block_size )
{
    int** TrB = transpose_matrix(B,N);
    int tmp = 0;
    int nthread = omp_get_num_threads();
    chunk = matrix_size/nthread;

        #pragma omp parallel shared(A,B,C,nthread, chunk) private(tmp)
            #pragma omp for schedule(dynamic,chunk)
            for(int i = 0; i < matrix_size/block_size; i++){
                for(int j = 0; j < matrix_size / block_size; j++){
                    for(int k = block_size*i; k< block_size*(i+1); k++){
                        for(int l = block_size*j; l< block_size*(j+1); l++){
                            tmp = 0;
                            for(int n = 0; n < matrix_size; n++)
                                    tmp += A[k][n]*TrB[l][n];
                            C[k][l] = tmp;
                        }
                    }
                }
            }
}

int main(int argc, char** argv) {

    int** A;
    int** B;
    int** C;

    //time measurement vars
    struct timeval  tv1, tv2;
    double run_time = 0.0;

    int naive_exec_time = 0;

    A = ones(N);
    B = ones(N);
	C = zeros(N);


    printf("General infos:\nBlock size: %d\nMatrix size: %d x %d\n",BLOCK_SIZE,N,N);

    //Naive matrix multiplication
    printf("Naive matrix Multiplication without optimization\n");

    gettimeofday(&tv1, NULL);

    multiply_matrix_naive(A,B,C,N);

    gettimeofday(&tv2, NULL);

    run_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
    printf("Time taken to complete: %f s\n",run_time);
    naive_exec_time = run_time;

    //transpose with omp
    printf("\nmatrix transpose optimization with OpenMP\n");

    free_matrix(C,N);
    C = zeros(N);

    gettimeofday(&tv1, NULL);

    multiply_matrix_transpose_omp(A,B,C,N);

    gettimeofday(&tv2, NULL);

    run_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
    printf("Time taken to complete: %f s\n",run_time);
	percentage = ((double)(naive_exec_time - run_time)/(double)naive_exec_time)*100;
	printf("Operation improvement percentage: %f%%\n",percentage);

    //2D tiling and transpose with omp
    printf("\n2D tiling and matrix transpose optimization with OpenMP\n");

    free_matrix(C,N);
    C = zeros(N);

    gettimeofday(&tv1, NULL);

    multiply_matrix_tiling_2d_with_transposition_omp(A,B,C,N,BLOCK_SIZE);

    gettimeofday(&tv2, NULL);

    run_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
    printf("Time taken to complete: %f s\n",run_time);
	percentage = ((double)(naive_exec_time - run_time)/(double)naive_exec_time)*100;
	printf("Operation improvement percentage: %f%%\n",percentage);

    //free all matrix
    free_matrix(A,N);
    free_matrix(B,N);
    free_matrix(C,N);

    return 0;

}
