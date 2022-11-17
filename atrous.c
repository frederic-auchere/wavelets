#include <stdio.h>
#include <stdlib.h>
#include<math.h> 

int atrous(double *image, int id1, int id2,
		  double* kernel, int kd,
		  double *bilateral_variance, int bd1, int bd2,
		  int s,
		  double *output)
{

  // output init by image multiply by center of kernel
  int kcenter;
  int half_k = kd/2; 
  kcenter = kd*half_k+half_k;
  fprintf(stderr,"kcenter %d %lf  half_k %d \n",kcenter, kernel[kcenter], half_k);
  //? for (int i=0; i < id1*id2; i++) output[i]= image[i]*kernel[kcenter];

  double * padded;
  int p1, p2, inc;
  inc = half_k*pow(2,s)*2;
  p1 = (id1 + inc);
  p2 = (id2 + inc);
  fprintf(stderr,"s = %d padded size (%d,%d) \n",s, p1,p2);
  padded = (double *) malloc(sizeof(double) * p1*p2);
  int hinc = inc/2;
  for(int j=0,y=hinc; j<p2; j++) {
    if (y<0 || y >=id2) {fprintf(stderr, "err %d %d \n",j,y);return(2);}
    for(int i=0,x=hinc; i<p1; i++) {
      //checks
      if (x<0 || x >=id1) {fprintf(stderr, "err %d %d %d %d\n",i,j,x,y);return(1);}
      //fprintf(stderr,"i %d j%d x %d y %d \n",i,j,x,y);
      padded[j*p1+i]= image[x*id1+y];  
      x += (i<hinc||(i+1)>=(hinc+id1) ? -1 : +1);
    }
    y += (j<hinc||(j+1)>=(hinc+id2) ? -1 : +1);
  }

  for(int j=0; j< id2; j++) {
    for (int i=0;i< id1; i++) {
      
    }
  }

  fprintf(stderr," %lf \n", image[2048]);

  
  return(0);
}

