#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <openacc.h>
#include <math.h>

#define BSIZE  16
#define NN 4000;
#define MM 6000;

void MatAdd(int N, int M, float *A, float *B, float *C) {
//#pragma acc kernels copyin(A[0:M*N],B[0:M*N]), copyout(C[0:M*N])
	for(int i=0; i<M; i++) {
	  for(int j=0; j<N; j++) {
	    C[i*N+j] = A[i*N+j] + B[i*N+j];
	  }
	}
}

 void err_exit(char *message);
 float mat_add_check(int n, float *x, float *y, float *z)  {
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
//-------------------- fill arrays A,B

 for (i=0; i<M; i++) 
    for (j=0; j<N; j++) {
      A[i*N+j] = (float) rand() / (float) rand();
      B[i*N+j] = (float) rand() / (float) rand();
} 
 t1 = wctime();
   MatAdd(N, M, A, B, C);
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
}

//-------------------- Prints error error Msg and exits 
void err_exit(char *errMsg) {
	printf("%s\n", errMsg);
	exit(1);
}
