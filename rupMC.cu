#include "rupMC.h"

#include "MCTriangleTable.h"
#include "MC.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <iostream>

#include <string>
#include <fstream>

int rupMC(int myid, float value, int readloop, int *Time, int *dims, double *origins, double *spacing, float *data, int *Loops, float*** PTS, int*** TRA, int** count, int* PTSTRAStartStop)
{
	time_t t[2] = { 0 };

	int C;
	cudaGetDeviceCount(&C);
	int GPUId = (myid - 2) % C;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(GPUId);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPUId);
	unsigned int globalmem = deviceProp.totalGlobalMem;

	size_t PerKSize = (dims[0] * dims[1] * (sizeof(float) + 4 * sizeof(int)) + (dims[0] - 1)*(dims[1] - 1)*(5 * 3 * 3 * sizeof(float) + 5 * 3 * sizeof(int)))*0.8;
	size_t PerStep = (globalmem - (3 + 8 + 256 * 16 + 12) * sizeof(int) - sizeof(float)) / PerKSize;
	*Loops = 1;
	if (PerStep < dims[2])
	{
		*Loops = dims[2] / PerStep + 1;
		PerStep = dims[2] / *Loops + 3;
	}

	*PTS = new float*[*Loops];
	*TRA = new int*[*Loops];
	*count = new int[*Loops * 2]{ 0 };
	int *d_dims;
	cudaMalloc((void**)&d_dims, 3 * sizeof(int));
	cudaMemcpy(d_dims, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);
	double *d_spacing, *d_origin;
	cudaMalloc((void**)&d_spacing, 3 * sizeof(double));
	cudaMemcpy(d_spacing, spacing, 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_origin, 3 * sizeof(double));
	cudaMemcpy(d_origin, origins, 3 * sizeof(double), cudaMemcpyHostToDevice);
	int *d_CASE_MASK, *d_MC_TRIANGLE_TABLE;
	cudaMalloc((void**)&d_CASE_MASK, 8 * sizeof(int));
	cudaMemcpy(d_CASE_MASK, CASE_MASK, 8 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_MC_TRIANGLE_TABLE, 256 * 16 * sizeof(int));
	cudaMemcpy(d_MC_TRIANGLE_TABLE, MC_TRIANGLE_TABLE, 256 * 16 * sizeof(int), cudaMemcpyHostToDevice);
	int edgeToIdx[12];
	edgeToIdx[0] = 0;
	edgeToIdx[1] = 4;
	edgeToIdx[2] = dims[0] * 3;
	edgeToIdx[3] = 1;
	edgeToIdx[4] = dims[0] * dims[1] * 3;
	edgeToIdx[5] = dims[0] * dims[1] * 3 + 4;
	edgeToIdx[6] = (dims[0] * dims[1] + dims[0]) * 3;
	edgeToIdx[7] = dims[0] * dims[1] * 3 + 1;
	edgeToIdx[8] = 2;
	edgeToIdx[9] = 5;
	edgeToIdx[10] = dims[0] * 3 + 2;
	edgeToIdx[11] = (dims[0] + 1) * 3 + 2;
	int *d_edgeToIdx;
	cudaMalloc((void**)&d_edgeToIdx, 12 * sizeof(int));
	cudaMemcpy(d_edgeToIdx, edgeToIdx, 12 * sizeof(int), cudaMemcpyHostToDevice);
	int PTSStartIndex = 0;

	if ((myid - 2) == 0)
	{
		PTSTRAStartStop[0] = 0;
		PTSTRAStartStop[2] = 0;
	}
	if ((myid - 2) == readloop - 1)
	{
		PTSTRAStartStop[1] = 1;
		PTSTRAStartStop[3] = 1;
	}

	for (int L = 0, StartK = 0; L < *Loops&&StartK < dims[2]; L++, StartK += PerStep - 3)
	{
		if (L == *Loops - 1)
			PerStep = dims[2] - StartK;

		int blkwidth = 16;
		int blkheight = 16;
		dim3 blocks(blkwidth, blkheight);
		dim3 grids((dims[0] - 1) % blkwidth == 0 ? (dims[0] - 1) / blkwidth : (dims[0] - 1) / blkwidth + 1, (dims[1] - 1) % blkheight == 0 ? (dims[1] - 1) / blkheight : (dims[1] - 1) / blkheight + 1);

		cudaMemcpy(&d_dims[2], &PerStep, sizeof(int), cudaMemcpyHostToDevice);
		float *d_data;
		cudaMalloc((void**)&d_data, sizeof(float) * dims[0] * dims[1] * PerStep);
		cudaMemcpy(d_data, &data[StartK*dims[0] * dims[1]], sizeof(float) * dims[0] * dims[1] * PerStep, cudaMemcpyHostToDevice);

		int *d_PTSMark;
		cudaMalloc((void**)&d_PTSMark, (dims[0] * dims[1] * PerStep) * 3 * sizeof(int));
		cudaMemset(d_PTSMark, 0, (dims[0] * dims[1] * PerStep) * 3 * sizeof(int));
		int *d_TRAMark;
		cudaMalloc((void**)&d_TRAMark, (dims[0] * dims[1] * PerStep) * sizeof(int));
		cudaMemset(d_TRAMark, 0, (dims[0] * dims[1] * PerStep) * sizeof(int));

		time(&t[0]);
		MCComputeMark << <grids, blocks >> > (d_dims, d_data, value, d_CASE_MASK, d_MC_TRIANGLE_TABLE, d_edgeToIdx, d_PTSMark, d_TRAMark);
		cudaDeviceSynchronize();
		time(&t[1]);
		*Time += t[1] - t[0];
		
		int* PTSMark = new int[dims[0] * dims[1] * PerStep * 3];
		int* TRAMark = new int[dims[0] * dims[1] * PerStep];
		cudaMemcpy(PTSMark, d_PTSMark, (dims[0] * dims[1] * PerStep) * 3 * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(TRAMark, d_TRAMark, (dims[0] * dims[1] * PerStep) * sizeof(int), cudaMemcpyDeviceToHost);
		int MarkStart = dims[0] * dims[1];
		int MarkStop = (PerStep - 1)*dims[0] * dims[1];
		if (L == 0)
			MarkStart = 0;
		if (L == *Loops - 1)
			MarkStop = PerStep*dims[0] * dims[1];
		for (int i = MarkStart * 3 + 1; i < MarkStop * 3; i++)
			PTSMark[i] += PTSMark[i - 1];
		for (int i = MarkStart + 1; i < MarkStop; i++)
			TRAMark[i] += TRAMark[i - 1];
		count[0][L * 2] = PTSMark[(MarkStop - dims[0] * dims[1]) * 3 - 1];
		count[0][L * 2 + 1] = TRAMark[(MarkStop - dims[0] * dims[1]) - 1];

		if (PTSTRAStartStop[0] != 0 && L == 0)
		{
			PTSTRAStartStop[0] = PTSMark[PTSTRAStartStop[0] * dims[0] * dims[1] * 3 - 1];
			PTSTRAStartStop[2] = TRAMark[PTSTRAStartStop[2] * dims[0] * dims[1] - 1];
		}
		if (L == *Loops - 1)
		{
			PTSTRAStartStop[1] = PTSMark[(MarkStop - PTSTRAStartStop[1] * dims[0] * dims[1]) * 3 - 1];
			PTSTRAStartStop[3] = TRAMark[(MarkStop - PTSTRAStartStop[3] * dims[0] * dims[1]) - 1];
		}

		int tempdims = PerStep - 2;
		if (L > 0 && L < *Loops - 1)
			tempdims = PerStep - 3;
		if (L == *Loops - 1)
			tempdims = PerStep - 1;
		cudaMemcpy(&d_dims[2], &tempdims, sizeof(int), cudaMemcpyHostToDevice);
		double origin = origins[2] + (StartK + MarkStart / (dims[0] * dims[1])) *spacing[2];
		cudaMemcpy(&d_origin[2], &origin, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_PTSMark, &PTSMark[MarkStart * 3], (MarkStop - MarkStart) * 3 * sizeof(int), cudaMemcpyHostToDevice);
		float *d_PTS;
		cudaMalloc((void**)&d_PTS, count[0][L * 2] * 3 * sizeof(float));
		cudaMemset(d_PTS, 0, count[0][L * 2] * 3 * sizeof(float));

		time(&t[0]);
		MCComputePoint << <grids, blocks >> > (d_dims, d_spacing, d_origin, &d_data[MarkStart],value, d_PTSMark, d_PTS);
		cudaDeviceSynchronize();
		time(&t[1]);
		*Time += t[1] - t[0];

		PTS[0][L] = new float[count[0][L * 2] * 3];
		cudaMemcpy(PTS[0][L], d_PTS, count[0][L * 2] * 3 * sizeof(float), cudaMemcpyDeviceToHost);

		tempdims = PerStep;
		if (L > 0 && L < *Loops - 1)
			tempdims = PerStep - 2;
		if (L == *Loops - 1)
			tempdims = PerStep - 1;
		cudaMemcpy(&d_dims[2], &tempdims, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_TRAMark, &TRAMark[MarkStart], (MarkStop - MarkStart) * sizeof(int), cudaMemcpyHostToDevice);
		int *d_TRA;
		cudaMalloc((void**)&d_TRA, count[0][L * 2 + 1] * 3 * sizeof(int));
		cudaMemset(d_TRA, 0, count[0][L * 2 + 1] * 3 * sizeof(int));

		time(&t[0]);
		MCComputeTriangle << <grids, blocks >> > (d_dims, &d_data[MarkStart], value,d_CASE_MASK, d_MC_TRIANGLE_TABLE, d_edgeToIdx, d_PTSMark, d_TRAMark, d_TRA, PTSStartIndex);
		cudaDeviceSynchronize();
		time(&t[1]);
		*Time += t[1] - t[0];

		TRA[0][L] = new int[count[0][L * 2 + 1] * 3];
		cudaMemcpy(TRA[0][L], d_TRA, count[0][L * 2 + 1] * 3 * sizeof(int), cudaMemcpyDeviceToHost);
		PTSStartIndex += count[0][L * 2];

		cudaFree(d_data);
		cudaFree(d_PTSMark);
		cudaFree(d_TRAMark);
		cudaFree(d_PTS);
		cudaFree(d_TRA);

		delete[] PTSMark;
		delete[] TRAMark;
	}
	cudaFree(d_dims);
	cudaFree(d_spacing);
	cudaFree(d_origin);
	cudaFree(d_CASE_MASK);
	cudaFree(d_MC_TRIANGLE_TABLE);
	cudaFree(d_edgeToIdx);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}


