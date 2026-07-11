//c语言实现动态形状的矩阵列相加乘法
#include <time.h>
#include<stdio.h>
#include <stdlib.h>
void matrix_multiply(int **A, int **B, int **C, int rowsB, int rowsA, int colsB) {
    for(int i=0;i<colsB;i++){
        for(int j=0;j<rowsB;j++){
            for(int k=0;k<rowsA;k++)
            {

                C[k][i]=B[j][i]*A[k][j]+C[k][i];
            }
        }
        }
}
void initialize_matrix(int ***matrix, int rows, int cols) {
        *matrix=malloc (sizeof(int*) * rows);
    for(int i =0;i<rows;i++){
        (*matrix)[i]=malloc (sizeof(int) * cols);
    }

    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            srand((unsigned int)time(NULL));
            (*matrix)[i][j]=rand()%10;
        }
    }
}
int main()
{
    int rowsA=1000, colsA=1000, rowsB=1000, colsB=1000;
    int **A, **B, **C;
    initialize_matrix(&A, rowsA, colsA);
    initialize_matrix(&B, rowsB, colsB);
    initialize_matrix(&C, rowsA, colsB);
   clock_t
 start = clock();
matrix_multiply(A, B, C, rowsB, rowsA, colsB);
clock_t
 end = clock();
double time_spent = (double
)(end - start) / CLOCKS_PER_SEC;
printf("time: %f s\n"
, time_spent);
    return 0;
}

