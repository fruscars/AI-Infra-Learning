//c语言实现动态形状的矩阵行列乘法
#include <time.h>
#include<stdio.h>
#include <stdlib.h>
void matrix_multiply(int **A, int **B, int **C, int rowsA, int colsA, int colsB) {
    // 将循环顺序调整为 i-k-j，这是优化性能的关键
    for (int i = 0; i < rowsA; i++) {
        for (int k = 0; k < colsA; k++) {
            int temp = A[i][k]; // 缓存 A 的值，减少重复访存
            for (int j = 0; j < colsB; j++) {
                C[i][j] += temp * B[k][j]; // 此时访问 B[k][j] 和 C[i][j] 都是连续内存访问
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
