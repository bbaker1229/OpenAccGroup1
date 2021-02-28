#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>

#define BSIZE  16
#define NN 4000;
#define MM 6000;

__global__ void MatAdd(int N, int M, float *A, float *B, float *C){ 
   int j = blockIdx.x * blockDim.x + threadIdx.x;
   int i = blockIdx.y * blockDim.y + threadIdx.y;
   if (i < M && j < N)
       C[i*N+j] = A[i*N+j] + B[i*N+j]; 
} 

 void err_exit(char *message);
 float mat_add_check(int n,  float *x, float *y, float *z)  {
 float s=0.0, t = 0.0, td = 0.0;
 for (int i=0; i<n; i++) {
       s  = y[i]+x[i]-z[i]; 
       t += s*s ;
       td += (x[i]*x[i]+y[i]*y[i]);
 }    

//-------------------- matrices are both zero
 if (td == 0.0) return(-1);
    else
//-------------------- normal return
   return(sqrt(s/td));
} 

double wctime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

int main() {
float *Ad, *Bd, *Cd; 
float  *A,  *B,  *C; 
int N, M, i, j; 
size_t MatSize;
float s;
double wctime();
double t1;
float nops;
//-------------------- set dimension N
 N = NN;
 M = MM;

 char LineG[] = "Error allocating GPU  memory";
 char LineH[] = "Error allocating Host memory";

  
 MatSize = N*M*sizeof(float);
//-------------------- allocate on cpu
 A = (float *)malloc(MatSize);        
 B = (float *)malloc(MatSize);        
 C = (float *)malloc(MatSize);    
 if ((A==NULL) | (B==NULL) | (C==NULL) ) 
          err_exit(LineH);
//-------------------- allocate on GPU
 if (cudaMalloc((void **) &Ad, MatSize) != cudaSuccess) 
       err_exit(LineG);
 if (cudaMalloc((void **) &Bd, MatSize) != cudaSuccess) 
       err_exit(LineG);
 if (cudaMalloc((void **) &Cd, MatSize) != cudaSuccess) 
       err_exit(LineG);
//-------------------- fill arrays A,B

 for (i=0; i<M; i++) 
    for (j=0; j<N; j++) {
      A[i*N+j] = (float) rand() / (float) rand();
      B[i*N+j] = (float) rand() / (float) rand();
} 
//
//-------------------- copy matrices A,B+ to GPU memory
t1 = wctime();
cudaMemcpy(Ad, A, MatSize, cudaMemcpyHostToDevice);
cudaMemcpy(Bd, B, MatSize, cudaMemcpyHostToDevice);
//-------------------- Kernel invocation
   dim3 dimBlock(BSIZE, 256/BSIZE);
// x: columns , y: rows    
   dim3 dimGrid((N + dimBlock.x-1) / dimBlock.x,
                (M + dimBlock.y-1) / dimBlock.y);
   MatAdd<<<dimGrid, dimBlock>>>(N, M, Ad, Bd, Cd);
//-------------------- see if things did execute 
 cudaError_t error = cudaGetLastError();
 if (error) {
     printf("CUDA error: %s \n",cudaGetErrorString(error));
     exit(1);
 }
//-------------------- Transfer result from GPU to CPU
cudaMemcpy(C, Cd, MatSize, cudaMemcpyDeviceToHost);
t1 = (wctime() - t1);
//-------------------- check whether addition was correct
s =  mat_add_check(N*M,A,B,C);
 
printf(" Mat dims M = %d  N = %d  -- err= %10.6e\n",M,N,s); 
printf(" Function runtime = %f seconds\n",t1);
t1 = t1 * 1.e+06;
nops = (float) M*N;
printf(" Performance = %f Mflops\n",nops/t1);
//-------------------- Free Host arrays
 free(A); 
 free(B);
 free(C);
//-------------------- Free GPU memory
 cudaFree(Ad);
 cudaFree(Bd);
 cudaFree(Cd);	
}

//-------------------- Prints error error Msg and exits 
void err_exit(char *errMsg) {
	printf("%s\n", errMsg);
	exit(1);
}
