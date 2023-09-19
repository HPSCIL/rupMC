#ifndef RUPMC_H
#define RUPMC_H

int rupMC(int myid, float value, int readloop, int *Time, int *dims, double *origins, double *spacing, float *data, int *Loops, float*** PTS, int*** TRA, int** count, int* PTSTRAStartStop);

#endif