/**
 * @file cpu.c
 * @brief Only CPU calculation to model the propagation of heat inside a cylindrical radiator.
 * 	Code for Task1 of MAP55616 Assignment2.
 * @author Luning
 * @version 1.0
 * @date 2023-04-17
 */

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdbool.h>
#include<sys/time.h>

/* function to print the radiator matrix */
void print_matrix(float **matrix, int n, int m){
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

int main(int argc, char **argv){
	int n=32;	// Matrix row size
	int m=32;	// Matrix col size
	int p=10;	// Iteration
	bool showAverage = false;	// Option to show average temperature
	bool displayTiming = false;	// Option to show the runtime for CPU version
	double t;

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
		else if (strcmp(argv[i], "-t") == 0){
			displayTiming = true;
		}
	}

	// Allocate matrices
	float **current_matrix = (float**)malloc(n * sizeof(float*));
	float **next_matrix = (float**)malloc(n * sizeof(float*));
	for(int i=0; i<n; i++){
		current_matrix[i] = (float*)malloc(m * sizeof(float));
		next_matrix[i] = (float*)malloc(m * sizeof(float));
	}

	// Initialize matrices
	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++){
			if(j==0){
				current_matrix[i][j] = 1.0f * (float)(i+1) / (float)(n);
				next_matrix[i][j] = 1.0f * (float)(i+1) / (float)(n);

			} else if(j==1){
				current_matrix[i][j] = 0.7f* (float)(i+1) / (float)(n);
				next_matrix[i][j] = 0.7f* (float)(i+1) / (float)(n);

			} else{
				current_matrix[i][j] = 1.0f / 15360.0f;
				next_matrix[i][j] = 1.0f / 15360.0f;
			}
		}
	}

	// Propagate heat
	t = get_time();
	for(int k=0; k<p; k++){
		for(int i=0; i<n; i++){
			for(int j=2; j<m; j++){
					next_matrix[i][j] = 1.65f * current_matrix[i][j-2] +
							    1.35f * current_matrix[i][j-1] +
							    	    current_matrix[i][j] +
							    0.65f * current_matrix[i][j+1] +
							    0.35f * current_matrix[i][(j+2)%m];
					next_matrix[i][j] /= 5.0f;
			}
		}

		// Swap the matrics
		float **tmp = current_matrix;
		current_matrix = next_matrix;
		next_matrix = tmp;
	}
	t = get_time()-t;

	// Print time
	if(displayTiming){
		printf("RunTime of CPU calculation is %lf s\n",t);
	}

	// Print result matrix(for test)
	//print_matrix(current_matrix, n, m);

	// Calculate the average temperature
	if(showAverage){
		for(int i=0; i<n; i++){
			float row_sum = 0.0f;
			for(int j=0; j<m; j++){
				row_sum += current_matrix[i][j];
			}
			float row_avg = row_sum / (float)m;
			printf("Row %d: %.6f\n", i, row_avg);
		}

	}

	free(current_matrix);
	free(next_matrix);

	return 0;

}
