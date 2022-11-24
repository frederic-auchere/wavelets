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
  //fprintf(stderr,"kcenter %d %lf  half_k %d \n",kcenter, kernel[kcenter], half_k);

  double * padded, *shifted, *weight, *norm;
  int p1, p2, inc;
  int pow_s;
  pow_s = pow(2,s);
  
  inc = half_k*pow_s*2;
  p1 = (id1 + inc);
  p2 = (id2 + inc);
  //fprintf(stderr,"s = %d pow_s %d inc %d padded size (%d,%d) \n",s, pow_s, inc, p1,p2);
  padded = (double *) malloc(sizeof(double) * p1*p2);
  shifted = (double *) malloc (sizeof(double) * id1*id2);
  weight = (double *) malloc  (sizeof(double) * id1*id2);
  norm = (double *) malloc  (sizeof(double) * id1*id2);
  
  int hinc = inc/2;
  for(int j=0; j<id2; j++) {
    for(int i=0; i<id1; i++) {
      padded[(j+hinc)*p1+(i+hinc)]= image[j*id1+i];  
    }
  }
  // maintenant les bords
  // SW
  for(int j=0;j<p2;j++) {
    for(int i=0;i<hinc;i++){
      //fprintf(stderr, " pad %d,%d = image %d,%d donne %lf\n",i,j,hinc-i-1,j,image[j*id1+(hinc-i-1)] );
      padded[j*p1+i] = padded[j*p1+(hinc+1-i)]; // en 1 je veux 2, en 0 3
    }
    for(int i=0;i<hinc;i++)
      padded[j*p1+(p1-hinc)+i] = padded[j*p1+(p1-hinc-1)-i];
  }
  for(int i=0;i<p1;i++) {
    for(int j=0;j<hinc;j++)
      padded[j*p1+i] = padded[(hinc+1-j)*p1+i]; 
    for(int j=0;j<hinc;j++)
      padded[(p2-hinc+j)*p1+i] = padded[(p2-hinc-1-j)*p1+i];  //2050, 2051, recoit 2049,2048
  }

  

  // bordel pour ecrire le fichier fits
  fitsfile *fptr;
  char filename[80], cmd[80];
  sprintf(filename, "padded_c_%d.fits",s);
  sprintf(cmd,"rm padded_c_%d.fits",s);
  system(cmd);
  int bitpix   =  -64;
  long naxis    =   2;
  long naxes[2] = { p1, p2 };
  int status = 0;
  fits_create_file(&fptr, filename, &status);
  fits_create_img(fptr,  bitpix, naxis, naxes, &status);
  status=0;
  if (fits_write_img(fptr, TDOUBLE, 1, p1*p2, padded, &status) ) fprintf(stderr,"write img status %d",status);
  fits_close_file(fptr, &status); 
  //debug
  sprintf(filename, "image_c_%d.fits",s);
  sprintf(cmd,"rm image_c_%d.fits",s);
  system(cmd);
  naxes[0] = id1; naxes[1]=id2;
  fits_create_file(&fptr, filename, &status);
  fits_create_img(fptr,  bitpix, naxis, naxes, &status);
  status=0;
  if (fits_write_img(fptr, TDOUBLE, 1, p1*p2, image, &status) ) fprintf(stderr,"write img status %d",status);
  fits_close_file(fptr, &status); 

  for(int j=0;j<id2;j++)
    for (int i=0;i<id1;i++){
      norm[j*id1+i]=kernel[kcenter];
      output[j*id1+i]=image[j*id1+i]*kernel[kcenter];
    }

  int l,m;
  for (int y= -half_k; y<= half_k; y++) {
    m = y*pow_s;
    for (int x = -half_k; x<= half_k; x++) {
      l = x*pow_s;
      for(int j=0; j< id2; j++) {
	for (int i=0;i< id1; i++) {
	  shifted[j*id1+i] = padded[((j+hinc)+m)*p1+(i+hinc)+l];
	}
      }
       for(int j=0; j< id2; j++) {
	for (int i=0;i< id1; i++) {
	  weight[j*id1+i] = kernel[(y+half_k)*kd+(x+half_k)]*
	    exp(-((image[j*id1+i]-shifted[j*id1+i])*(image[j*id1+i]-shifted[j*id1+i]))/bilateral_variance[j*id1+i]/2);
	}
      }
       for(int j=0; j< id2; j++) {
	for (int i=0;i< id1; i++) {
	  norm[j*id1+i] +=  weight[j*id1+i];
	}
       }
       for(int j=0; j< id2; j++) {
	for (int i=0;i< id1; i++) {
	  output[j*id1+i] = (output[j*id1+i] + (shifted[j*id1+i]*weight[j*id1+i]));
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

