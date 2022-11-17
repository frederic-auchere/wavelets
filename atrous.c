#include <stdio.h>

int atrous(double *image, int id1, int id2,
		  double* kernel, int kd1, int kd2,
		  double *bilateral_variance, int bd1, int bd2,
		  int s,
		  char *,
		  double *output)
{

  // output init by image multiply by center of kernel
  int kcenter;
  kcenter = kd1*(kd2/2)+kd1/2;
  fprintf(stderr,"kcenter %d %lf\n",kcenter, *kernel);
  
  

  //fprintf(stderr," %lf ", image[2048]);

  //for (int i=0; i < id1*id2; i++) output[i]= image[i]*kernel[kcenter];
  
  return(0);
}

