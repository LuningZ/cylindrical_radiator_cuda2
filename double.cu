/**
 * @file double.c
 * @brief Parallel implementation (Double-precision) to model the propagation of heat inside a cylindrical radiator.
 * 	Code for Task2 of MAP55616 Assignment2.
 * @author Luning
 * @version 1.0
 * @date 2023-04-17
 */

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdbool.h>
#include<sys/time.h>
#include<cuda_runtime.h>

#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 16

void print_matrix(double **matrix, int n, int m);
double get_time(void);

__global__ void init_matrices_kernel(double* d_matrix, int n, int m){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i<n && j<m){
		int index = i*m+j;
		if(j==0){
			d_matrix[index] = 1.0 * (double)(i+1) / (double)(n);
		} else if(j==1){
			d_matrix[index] = 0.7 * (double)(i+1) / (double)(n);
		} else{
			d_matrix[index] = 1.0 / 15360.0;
		}
	}
}

__global__ void propagate_heat_kernel(double* current_matrix, double* next_matrix, int n, int m){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i>=n || j>=m)
		return;	// Out of bounds

	if(j<2){
		next_matrix[i*m+j] = current_matrix[i*m+j];
	}
	else{
		next_matrix[i*m+j] = 1.65 * current_matrix[i*m+((j-2)+m)%m] +
				     		1.35 * current_matrix[i*m+((j-1)+m)%m] +
					    			current_matrix[i*m+j] +
				     		0.65 * current_matrix[i*m+(j+1)%m] +
					 		0.35 * current_matrix[i*m+(j+2)%m];
		next_matrix[i*m+j] /= 5.0;	
	}

}

__global__ void swap_matrices_kernel(double *d_current, double *d_next, int n, int m){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i<n && j<m){
		int index = i*m+j;
		double temp = d_current[index];
		d_current[index] = d_next[index];
		d_next[index] = temp;
	}

}

__global__ void calculate_average_kernel(double* current_matrix, double* row_avg, int n, int m){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n){
		double row_sum = 0.0;
		for(int j=0; j<m; j++){
			row_sum += current_matrix[i*m+j];
		}
		row_avg[i] = row_sum/(double)m;
	}
}

int main(int argc, char **argv){
	int n=32;	// Matrix row size
	int m=32;	// Matrix col size
	int p=10;	// Iteration
	bool showAverage = false;	// Option to show average temperature
	bool skipCpu = false; 		// Option to skip CPU calculation
	bool displayTiming = false;	// Option to display both the CPU and GPU timings
	double t_cpu = 0.0;
	float t_gpu_allocate = 0.0, t_gpu_compute = 0.0, t_gpu_average = 0.0, t_gpu_transfer = 0.0;

	// Parse command line arguments
	for(int i=1; i<argc; i++){
		if (strcmp(argv[i], "-n") == 0 && i+1 < argc){
			n = atoi(argv[i+1]);
		}
		else if (strcmp(argv[i], "-m") == 0 && i+1 < argc){
			m = atoi(argv[i+1]);
		}
		else if (strcmp(argv[i], "-p") == 0 && i+1 < argc){
			p = atoi(argv[i+1]);
		}
		else if (strcmp(argv[i], "-a") == 0){
			showAverage = true;
		}
		else if (strcmp(argv[i], "-c") == 0){
			skipCpu = true;
		}
		else if (strcmp(argv[i], "-t") == 0){
			displayTiming = true;
		}
	}
	
	/* Only CPU calculation ============================================ */
    if(!skipCpu){
		// Allocate matrices
		double **current_matrix = (double**)malloc(n * sizeof(double*));
		double **next_matrix = (double**)malloc(n * sizeof(double*));
		for(int i=0; i<n; i++){
			current_matrix[i] = (double*)malloc(m * sizeof(double));
			next_matrix[i] = (double*)malloc(m * sizeof(double));
		}

		// Initialize matrices
		for(int i=0; i<n; i++){
			for(int j=0; j<m; j++){
				if(j==0){
					current_matrix[i][j] = 1.0 * (double)(i+1) / (double)(n);
					next_matrix[i][j] = 1.0 * (double)(i+1) / (double)(n);
				} else if(j==1){
					current_matrix[i][j] = 0.7 * (double)(i+1) / (double)(n);
					next_matrix[i][j] = 0.7 * (double)(i+1) / (double)(n);
				} else{
					current_matrix[i][j] = 1.0 / 15360.0;
					next_matrix[i][j] = 1.0 / 15360.0;
				}
			}
		}

		// Propagate heat (CPU calculation)
		t_cpu = get_time();
		for(int k=0; k<p; k++){
			for(int i=0; i<n; i++){
				for(int j=2; j<m; j++){
					next_matrix[i][j] = 1.65 * current_matrix[i][j-2] +
							    		1.35 * current_matrix[i][j-1] +
							    	    current_matrix[i][j] +
							    		0.65 * current_matrix[i][j+1] +
							    		0.35 * current_matrix[i][(j+2)%m];
					next_matrix[i][j] /= 5.0;
				}
			}

			// Swap the matrics
			double **tmp = current_matrix;
			current_matrix = next_matrix;
			next_matrix = tmp;
		}
		t_cpu = get_time() - t_cpu;
		//print_matrix(current_matrix, n, m);	// test

		// Calculate the average temperature
		if(showAverage){
			printf("Average temperature for each row: (CPU Calculation)\n");
			for(int i=0; i<n; i++){
				double row_sum = 0.0;
				for(int j=0; j<m; j++){
					row_sum += current_matrix[i][j];
				}
				double row_avg = row_sum / (double)m;
				printf("Row %d: %.6f\n", i, row_avg);
			}
		}
    
		free(current_matrix);
		free(next_matrix);
    }

    /* Parallel calculation ============================================ */
	dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 gridDim((n+blockDim.x-1) / blockDim.x, (m+blockDim.y-1) / blockDim.y);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// GPU allocation
	double *d_current, *d_next, *d_row_avg;
	cudaEventRecord(start);
	cudaMalloc(&d_current, n*m*sizeof(double));
	cudaMalloc(&d_next, n*m*sizeof(double));
	cudaMalloc(&d_row_avg, n*sizeof(double));
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&t_gpu_allocate, start, stop);

	init_matrices_kernel<<<gridDim,blockDim>>>(d_current, n, m);

	// Propagate heat on the GPU
	cudaEventRecord(start);
	for(int k=0; k<p; k++){
		propagate_heat_kernel<<<gridDim,blockDim>>>(d_current, d_next, n, m);
		cudaDeviceSynchronize();
		swap_matrices_kernel<<<gridDim,blockDim>>>(d_current, d_next, n, m);
		cudaDeviceSynchronize();
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&t_gpu_compute, start, stop);

	// Calculate the average tempreature
	cudaEventRecord(start);
	calculate_average_kernel<<<(n+blockDim.x-1)/blockDim.x, blockDim.x>>>(d_current, d_row_avg, n, m);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&t_gpu_average, start, stop);

	// Transfer back to the RAM 
	double *row_avg_gpu = (double *)malloc(n*sizeof(double));
	cudaEventRecord(start);
	cudaMemcpy(row_avg_gpu, d_row_avg, n*sizeof(double), cudaMemcpyDeviceToHost);
 	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&t_gpu_transfer, start, stop);


	// Display the timing results
	if(displayTiming){
		if(!skipCpu){ 
			printf("CPU time: %f s\n", t_cpu); 
		}
		printf("GPU time: \n");
		printf("  Allocation time: %f s\n", t_gpu_allocate/1000);
		printf("  Compute time: %.8f s\n", t_gpu_compute/1000);
		printf("  Calculate averages time: %f s\n", t_gpu_average/1000);
		printf("  Transfer time: %f s\n", t_gpu_transfer/1000);
	}

	// Print row averages
	if(showAverage) {
		printf("Average temperature for each row: (GPU Calculation)\n");
		for(int i=0; i<n; i++){
			printf("Row %d: %.6f\n", i, row_avg_gpu[i]);
		}
	}

	cudaFree(d_current);
	cudaFree(d_next);
	cudaFree(d_row_avg);
	free(row_avg_gpu);

	return 0;
}

/* function to print the radiator matrix */
void print_matrix(double **matrix, int n, int m){
	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++){
			printf("%.6f ", matrix[i][j]);
		}
		printf("\n");
	}
}

double get_time(void){
	struct timeval tv;
	double t;
	gettimeofday(&tv, NULL);
	t = tv.tv_sec + (double)tv.tv_usec * 1e-6;

	return t;
}

