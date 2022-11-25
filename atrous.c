#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "fitsio.h"

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

  double *padded, *norm;
  int p1, p2, inc;
  int pow_s;
  pow_s = pow(2,s);
  
  inc = half_k*pow_s*2;
  p1 = (id1 + inc);
  p2 = (id2 + inc);
  padded = (double *) malloc(sizeof(double) * p1*p2);
  norm = (double *) malloc  (sizeof(double) * id1*id2);
  
  int hinc = inc/2;
  for(int j=0; j<id2; j++) {
    for(int i=0; i<id1; i++) {
      padded[(j+hinc)*p1+(i+hinc)]= image[j*id1+i];  
    }
  }
  // maintenant les bords
  for(int j=0;j<p2;j++) {
    for(int i=0;i<hinc;i++){
      padded[j*p1+i] = padded[j*p1+(hinc*2-1-i)]; 
    }
    for(int i=0;i<hinc;i++){
      padded[j*p1+(p1-hinc)+i] = padded[j*p1+(p1-hinc-1)-i];
    }
  }
  for(int i=0;i<p1;i++) {
    for(int j=0;j<hinc;j++){
      padded[j*p1+i] = padded[(hinc*2-1-j)*p1+i];
    }
    for(int j=0;j<hinc;j++){
      padded[(p2-hinc+j)*p1+i] = padded[(p2-hinc-1-j)*p1+i];
    }
  }

  for(int j=0;j<id2;j++)
    for (int i=0;i<id1;i++){
      norm[j*id1+i]=kernel[kcenter];
      output[j*id1+i]=image[j*id1+i]*kernel[kcenter];
    }

  int l,m; double shifted_s,weight_s;
  for (int y= half_k; y>= -half_k; y--) {
    m = y*pow_s;
    for (int x = half_k; x>= -half_k; x--) {
      if (x == 0 && y == 0) continue;
      l = x*pow_s;
      for(int j=0; j< id2; j++) {
	for (int i=0;i< id1; i++) {
	  shifted_s = padded[((j+hinc)+m)*p1+(i+hinc)+l];
	  weight_s = kernel[(y+half_k)*kd+(x+half_k)]* exp(-((image[j*id1+i]-shifted_s)*(image[j*id1+i]-shifted_s))/bilateral_variance[j*id1+i]/2);
	  norm[j*id1+i] +=  weight_s;
	  output[j*id1+i] = output[j*id1+i] + (shifted_s*weight_s);
	}
      }
    }
  }

  for(int j=0; j< id2; j++) {
    for (int i=0;i< id1; i++) {
      output[j*id1+i] /= norm[j*id1+i]; 
    }
  }

  
  return(0);
}

