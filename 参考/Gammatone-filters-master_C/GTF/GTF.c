//
//  gtf.c
//  all-pole gammatone filters
//
//  Created by songtao on 2019/8/17.
//  Copyright © 2019 songtao. All rights reserved.
//

#include <stdio.h>
#include<math.h>
#include <stdlib.h>
#include <string.h>
#define MIN_VALUE 1e-20
#define pi 3.14159265358979323846

/* multiply b to a*/
void complex_multiply(double a[2], double b[2]){
  double tmp[2];
  tmp[0]=a[0]; tmp[1]=a[1];
  a[0] = tmp[0]*b[0]-tmp[1]*b[1];
  a[1] = tmp[0]*b[1]+tmp[1]*b[0];
}


void free_mem(double* ptr){
  free(ptr);
}


double* GTF(double*y, double*x,int x_len,int fs, double*cfs, double*bws, int n_band,
            int is_env_aligned, int is_fine_aligned, int delay_common,
            int is_gain_norm){

    int* delays = (int*)malloc(sizeof(int)*n_band); // for aligning
    double phi0,phi0_complex[2];
    double gain_band = 1; //
    int max_delay=0,delay_band=0;

    // variables for frequency shiftor
    double freq_shiftor[2],freq_shiftor_pre[2];
    double freq_shiftor_step[2];

    // variables related to filters
    double tpt = 2*pi*(1.0/(double)fs); // constant
    double p[5][2]; // IIR filter status
    double a[5]; // IIR filter coefficients
    double u0[2]; // FIR filter output
    double b[4]; // FIR filter coefficients
    double k; // constants for each band

    int band_i, sample_i, order;

    // FILE* logger = fopen("log.txt", "w");
    // fprintf(logger, "%d %d %d %d", is_env_aligned, is_fine_aligned, delay_common, is_gain_norm);
    // fclose(logger);
    
    // calculate delay of each band
    if(is_env_aligned == 1){
      max_delay = 0;
      for(band_i=0;band_i<n_band;band_i++){
        delays[band_i] = round(3.0/(2.0*pi*bws[band_i])*fs)/fs;  // integer samples
        if(delays[band_i]>max_delay){ // find maximum delays
          max_delay = delays[band_i];
        }
      }
      if(delay_common<0){ // if delay_common<0,
        delay_common = max_delay; // align all filters to maximum delay
      }
    }
    else{
        for(band_i=0;band_i<n_band;band_i++){
            delays[band_i]=0;
        }
        delay_common = 0;
    }

    memset(y,0,sizeof(double)*x_len*n_band);
    for(band_i=0;band_i<n_band;band_i++){
        // initiation of filter states
        memset(p,0,sizeof(double)*10);

        k = exp(-tpt*bws[band_i]);
        // filter coefficients
        a[0] = 1; a[1] = 4*k; a[2] = -6*k*k; a[3]=4*pow(k,3); a[4]=-pow(k, 4);
        b[0] = 1; b[1] = 1;   b[2] = 4*k;    b[3] = k*k;

        // filter amp gain normalized
        if(is_gain_norm==1){
          gain_band = pow(1-k,4)/(1+4*k+k*k)*2;
        }
        else{
          gain_band = pow(1.0/fs,3);
        }
        // aligned filter ir fine-structure
        if(is_fine_aligned==1){
          phi0 = -3.0*cfs[band_i]/bws[band_i];
        }
        else{
          phi0 = 0;
        }
        phi0_complex[0] = cos(phi0);
        phi0_complex[1] = sin(phi0);

        /*
        computation acceleration
        convert cos(\phi1+\ph2) and sin(\phi1+\phi2) into mutliplication
        and summation
        */
        freq_shiftor_step[0] = cos(tpt*cfs[band_i]);
        freq_shiftor_step[1] = -sin(tpt*cfs[band_i]);

        freq_shiftor_pre[0] = freq_shiftor_step[0];
        freq_shiftor_pre[1] = -freq_shiftor_step[1];

        delay_band = delays[band_i]-delay_common;
        for(sample_i=0; sample_i<x_len+delay_band; sample_i++){
            freq_shiftor[0] = (freq_shiftor_pre[0]*freq_shiftor_step[0] 
			    - freq_shiftor_pre[1]*freq_shiftor_step[1]);
            freq_shiftor[1] = (freq_shiftor_pre[1]*freq_shiftor_step[0] 
			    + freq_shiftor_pre[0]*freq_shiftor_step[1]);
            freq_shiftor_pre[0] = freq_shiftor[0];
            freq_shiftor_pre[1] = freq_shiftor[1];
	    
            // denominator part of filter equation
            // equivalent to add zeros to the end of x
            if(sample_i>=x_len){
                p[0][0] = 0;
                p[0][1] = 0;
            }
            else{
		p[0][0] = x[sample_i]*freq_shiftor[0];
		p[0][1] = x[sample_i]*freq_shiftor[1];
                complex_multiply(p[0], phi0_complex);
            }

            for(order=1;order<=4;order++){
                p[0][0] = p[0][0]+a[order]*p[order][0];
                p[0][1] = p[0][1]+a[order]*p[order][1];
            }

            // numerator part of filter equation
            u0[0] = 0; u0[1] = 0;
            for(order=1;order<=3;order++){
                u0[0] = u0[0]+b[order]*p[order][0];
                u0[1] = u0[1]+b[order]*p[order][1];
            }

            // final output = real part of filte result
            if(sample_i>=delay_band){
                y[band_i*x_len+sample_i-delay_band] = gain_band*(u0[0]*freq_shiftor[0]+u0[1]*freq_shiftor[1]);
		// fprintf(logger, "%f\n", y[band_i*x_len+sample_i-delay_band]);
            }
            // update filter states
            for(order=4;order>=1;order--){
                p[order][0] = p[order-1][0];
                p[order][1] = p[order-1][1];
            }
        }
    }

    free(delays);
    return y;
}
