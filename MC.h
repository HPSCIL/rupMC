#ifndef MC_H
#define MC_H

__global__ void MCComputeMark(int dims[3], float *scalars, float value, int *d_CASE_MASK, int *d_MC_TRIANGLE_TABLE, int *edgeToIdx, int *PTSMark,int *TRAMark);
__global__ void MCComputePoint(int dims[3], double spacing[3], double origin[3], float *scalars, float value, int *PTSMark, float *PTS);
__global__ void MCComputeTriangle(int dims[3], float *scalars,float value,int *CASE_MASK, int *MC_TRIANGLE_TABLE, int *edgeToIdx, int *PTSMark, int *TRAMark, int *TRA,int PTSStart);

#endif