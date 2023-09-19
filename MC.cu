#include "MC.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void MCComputeMark(int dims[3], float* scalars, float value, int* CASE_MASK, int* MC_TRIANGLE_TABLE, int* edgeToIdx, int* PTSMark, int* TRAMark)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= dims[0] - 1)
		return;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j >= dims[1] - 1)
		return;

	float s[8];
	for (int k = 0; k < (dims[2] - 1); k++)
	{
		s[0] = scalars[i + j * dims[0] + k * dims[0] * dims[1]];
		s[1] = scalars[i + 1 + j * dims[0] + k * dims[0] * dims[1]];
		s[2] = scalars[i + 1 + (j + 1) * dims[0] + k * dims[0] * dims[1]];
		s[3] = scalars[i + (j + 1) * dims[0] + k * dims[0] * dims[1]];
		s[4] = scalars[i + j * dims[0] + (k + 1) * dims[0] * dims[1]];
		s[5] = scalars[i + 1 + j * dims[0] + (k + 1) * dims[0] * dims[1]];
		s[6] = scalars[i + 1 + (j + 1) * dims[0] + (k + 1) * dims[0] * dims[1]];
		s[7] = scalars[i + (j + 1) * dims[0] + (k + 1) * dims[0] * dims[1]];

		if ((s[0] < value && s[1] < value && s[2] < value && s[3] < value && s[4] < value && s[5] < value && s[6] < value && s[7] < value) ||
			(s[0] > value && s[1] > value && s[2] > value && s[3] > value && s[4] > value && s[5] > value && s[6] > value && s[7] > value))
			continue;

		int index = 0;
		for (int ii = 0; ii < 8; ii++)
			if (s[ii] >= value)
				index |= CASE_MASK[ii];

		if (index == 0 || index == 255)
			continue;

		const int* edge = &MC_TRIANGLE_TABLE[index * 16];
		for (; edge[0] > -1; edge += 3)
		{
			for (int ii = 0; ii < 3; ii++)
				PTSMark[(i + j * dims[0] + k * dims[0] * dims[1]) * 3 + edgeToIdx[edge[ii]]] = 1;

			TRAMark[i + j * dims[0] + k * dims[0] * dims[1]] += 1;
		}
	}
}


__global__ void MCComputePoint(int dims[3], double spacing[3], double origin[3], float *scalars,float value, int *PTSMark, float *PTS)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= dims[0])
		return;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j >= dims[1])
		return;

	float pts[3], s[4];
	int ptIdx[4];
	for (int k = 0; k < dims[2]; k++)
	{
		pts[0] = origin[0] + i * spacing[0];
		pts[1] = origin[1] + j * spacing[1];
		pts[2] = origin[2] + k * spacing[2];

		s[0] = scalars[(i + j * dims[0] + k * dims[0] * dims[1])];
		s[1] = scalars[(i + j * dims[0] + k * dims[0] * dims[1]) + 1];
		s[2] = scalars[(i + j * dims[0] + k * dims[0] * dims[1]) + dims[0]];
		s[3] = scalars[(i + j * dims[0] + k * dims[0] * dims[1]) + dims[0] * dims[1]];

		ptIdx[0] = ((i + j * dims[0] + k * dims[0] * dims[1]) == 0 ? -1 : PTSMark[(i + j * dims[0] + k * dims[0] * dims[1]) * 3 - 1] - 1);
		ptIdx[1] = PTSMark[(i + j * dims[0] + k * dims[0] * dims[1]) * 3] - 1;
		ptIdx[2] = PTSMark[(i + j * dims[0] + k * dims[0] * dims[1]) * 3 + 1] - 1;
		ptIdx[3] = PTSMark[(i + j * dims[0] + k * dims[0] * dims[1]) * 3 + 2] - 1;

		float t = 0;

		if ((ptIdx[1] - ptIdx[0]) == 1)
		{
			s[0] = scalars[(i + j * dims[0] + k * dims[0] * dims[1])];
			s[1] = scalars[(i + j * dims[0] + k * dims[0] * dims[1]) + 1];
			t = (value - s[0]) / (s[1] - s[0]);
			PTS[ptIdx[1] * 3] = pts[0] + t * spacing[0];
			PTS[ptIdx[1] * 3 + 1] = pts[1];
			PTS[ptIdx[1] * 3 + 2] = pts[2];
		}
		if ((ptIdx[2] - ptIdx[1]) == 1)
		{
			s[0] = scalars[(i + j * dims[0] + k * dims[0] * dims[1])];
			s[2] = scalars[(i + j * dims[0] + k * dims[0] * dims[1]) + dims[0]];
			t = (value - s[0]) / (s[2] - s[0]);
			PTS[ptIdx[2] * 3] = pts[0];
			PTS[ptIdx[2] * 3 + 1] = pts[1] + t * spacing[1];
			PTS[ptIdx[2] * 3 + 2] = pts[2];
		}
		if ((ptIdx[3] - ptIdx[2]) == 1)
		{
			s[0] = scalars[(i + j * dims[0] + k * dims[0] * dims[1])];
			s[3] = scalars[(i + j * dims[0] + k * dims[0] * dims[1]) + dims[0] * dims[1]];
			t = (value - s[0]) / (s[3] - s[0]);
			PTS[ptIdx[3] * 3] = pts[0];
			PTS[ptIdx[3] * 3 + 1] = pts[1];
			PTS[ptIdx[3] * 3 + 2] = pts[2] + t * spacing[2];
		}
	}
}

__global__ void MCComputeTriangle(int dims[3], float *scalars, float value, int *CASE_MASK, int *MC_TRIANGLE_TABLE, int *edgeToIdx, int *PTSMark, int *TRAMark, int *TRA,int PTSStart)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= dims[0] -1)
		return;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j >= dims[1] -1)
		return;

	float s[8];
	for (int k = 0; k < dims[2] - 1; k++)
	{
		s[0] = scalars[i + j * dims[0] + k * dims[0] * dims[1]];
		s[1] = scalars[i + 1 + j * dims[0] + k * dims[0] * dims[1]];
		s[2] = scalars[i + 1 + (j + 1) * dims[0] + k * dims[0] * dims[1]];
		s[3] = scalars[i + (j + 1) * dims[0] + k * dims[0] * dims[1]];
		s[4] = scalars[i + j * dims[0] + (k + 1) * dims[0] * dims[1]];
		s[5] = scalars[i + 1 + j * dims[0] + (k + 1) * dims[0] * dims[1]];
		s[6] = scalars[i + 1 + (j + 1) * dims[0] + (k + 1) * dims[0] * dims[1]];
		s[7] = scalars[i + (j + 1) * dims[0] + (k + 1) * dims[0] * dims[1]];

		if ((s[0] < value && s[1] < value && s[2] < value && s[3] < value && s[4] < value && s[5] < value && s[6] < value && s[7] < value) ||
			(s[0] > value && s[1] > value && s[2] > value && s[3] > value && s[4] > value && s[5] > value && s[6] > value && s[7] > value))
			continue;

		int index = 0;
		for (int ii = 0; ii < 8; ii++)
			if (s[ii] >= value)
				index |= CASE_MASK[ii];

		if (index == 0 || index == 255)
			continue;

		const int* edge = &MC_TRIANGLE_TABLE[index * 16];
		int n = 0;
		for (; edge[0] > -1; edge += 3)
		{
			for (int ii = 0; ii < 3; ii++)
			{
				int triIdx;
				if ((i + j * dims[0] + k * dims[0] * dims[1]) != 0)
					triIdx = (TRAMark[i + j * dims[0] + k * dims[0] * dims[1] - 1] + n) * 3 + ii;
				else
					triIdx = n * 3 + ii;

				TRA[triIdx] = PTSMark[(i + j * dims[0] + k * dims[0] * dims[1]) * 3 + edgeToIdx[edge[ii]]] + PTSStart - 1;
			}
			n++;
		}
	}
}