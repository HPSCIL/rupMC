//

#include "rupMC.h"

#include <mpi.h>

#include <string>
#include <time.h>
#include <fstream>
#include <vector>
#include <stdlib.h>
using namespace std;

int main(int argc, char* argv[])
{
	int myid, numprocs;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	for (int file_i = 1; file_i <6 ; file_i++)
	{
		string infilePath = "D:/rupMC/Data/geomodel_" + to_string(file_i) + ".vtk";

		float values[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21};

		int readloop = numprocs - 2;

		int* Time = new int[4* numprocs] { 0 };//(Reading;Transfering;Writing;)
		time_t T[2] = { 0 };
		int globaldims[3];
		long long scalarsize = 0;
		double spacing[3];
		double globalorigins[3];
		float* data = NULL;
		int numValues = sizeof(values) / sizeof(values[0]);
		vector<float*> globalPTS;
		vector<int*> globalTRA;
		vector<int> CountPTS;
		vector<int> CountPTSStart;
		vector<int> CountTRA;
		int sizePTS = 0, sizeTRA = 0;

		time_t t[2] = { 0 };
		int dims[3];
		double origins[3];
		time(&T[0]);

		ifstream ifile(infilePath.c_str());
		if (myid == 0)
		{
			while (!ifile.eof())
			{
				string stmp;
				ifile >> stmp;
				if (stmp == "DIMENSIONS")
				{
					for (int i = 0; i < 3; i++)
						ifile >> globaldims[i];
					scalarsize = globaldims[0] * globaldims[1] * globaldims[2];
					data = new float[globaldims[0] * globaldims[1] * globaldims[2]];
				}
				else if (stmp == "ASPECT_RATIO")
				{
					for (int i = 0; i < 3; i++)
						ifile >> spacing[i];
				}
				else if (stmp == "ORIGIN")
				{
					for (int i = 0; i < 3; i++)
						ifile >> globalorigins[i];
				}
				else if (stmp == "LOOKUP_TABLE")
				{
					ifile >> stmp;
					break;
				}
			}
		}
		MPI_Bcast(spacing, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&numValues, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(values, numValues, MPI_FLOAT, 0, MPI_COMM_WORLD);
		time(&t[0]);

		int PTS_i = 0;
		int TRA_i = 0;
		int PTSStart_i = 0;
		for (int read_i = 0; read_i < readloop; read_i++)
		{
			if (myid == 0)
			{
				int readmeank = globaldims[2] / readloop + 1;
				if (read_i == readloop - 1)
					readmeank = globaldims[2] - read_i * readmeank;
				for (int i = 0; i < globaldims[0] * globaldims[1] * readmeank; i++)
				{
					if (!ifile.eof())
						ifile >> data[i + read_i * globaldims[0] * globaldims[1] * (globaldims[2] / readloop + 1)];
				}
				if (read_i == readloop - 1)
					ifile.close();
				time(&t[1]);
				Time[4 * myid] += t[1] - t[0];
				t[0] = t[1];
				int i = 2 + read_i;
				MPI_Send(&t[0], 1, MPI_INT, read_i + 2, 10000 + i, MPI_COMM_WORLD);//序号+来源*2+目的*2

				int tempdims[3];
				tempdims[0] = globaldims[0];
				tempdims[1] = globaldims[1];
				if (read_i == 0)
					tempdims[2] = readmeank;
				else
					tempdims[2] = readmeank + 3;

				double temporigins[3];
				temporigins[0] = globalorigins[0];
				temporigins[1] = globalorigins[1];
				temporigins[2] = globalorigins[2];
				if (read_i > 0)
					temporigins[2] = globalorigins[2] + (read_i * (globaldims[2] / readloop + 1) - 3) * spacing[2];

				MPI_Send(tempdims, 3, MPI_INT, i, 20000 + i, MPI_COMM_WORLD);
				MPI_Send(temporigins, 3, MPI_DOUBLE, i, 30000 + i, MPI_COMM_WORLD);
				unsigned long size = tempdims[0] * tempdims[1] * tempdims[2];
				if (read_i == 0)
					MPI_Send(data, size, MPI_FLOAT, i, 40000 + i, MPI_COMM_WORLD);
				else
					MPI_Send(data + (read_i * (globaldims[2] / readloop + 1) - 3) * tempdims[0] * tempdims[1], size, MPI_FLOAT, i, 40000 + i, MPI_COMM_WORLD);

				time(&t[1]);
				Time[1 + 4 * myid] += t[1] - t[0];
				t[0] = t[1];

				if (read_i == readloop - 1)
					MPI_Send(&Time[4 * myid], 4, MPI_INT, 1, 50001, MPI_COMM_WORLD);
			}
		}

		for (int num = 0; num < numValues; num++)
		{
			if (myid != 0 && myid != 1)
			{
				if (num == 0)
				{
					MPI_Recv(&t[0], 1, MPI_INT, 0, 10000 + myid, MPI_COMM_WORLD, &status);

					MPI_Recv(dims, 3, MPI_INT, 0, 20000 + myid, MPI_COMM_WORLD, &status);

					MPI_Recv(origins, 3, MPI_DOUBLE, 0, 30000 + myid , MPI_COMM_WORLD, &status);

					data = new float[dims[0] * dims[1] * dims[2]];
					MPI_Recv(data, dims[0] * dims[1] * dims[2], MPI_FLOAT, 0, 40000 + myid , MPI_COMM_WORLD, &status);

					time(&t[1]);
					Time[1+ 4 * myid] += t[1] - t[0];
					t[0] = t[1];
				}

				int comptime = 0;
				int Loops = 0;
				float** PTS;
				int** TRA;
				int* count;
				int PTSTRAStartStop[4] = { 1,2,1,2 };
				rupMC(myid, values[num], readloop, &comptime, dims, origins, spacing, data, &Loops, &PTS, &TRA, &count, PTSTRAStartStop);

				time(&t[1]);
				Time[2 + 4 * myid] += comptime;
				Time[1 + 4 * myid]+=t[1] - t[0]- comptime;
				t[0] = t[1];

				MPI_Send(&t[0], 1, MPI_INT, 1, 1000100 + myid * 10000 + num, MPI_COMM_WORLD);//序号+来源*2+目的*2+等值*2

				MPI_Send(&Loops, 1, MPI_INT, 1, 2000100 + myid * 10000 + num, MPI_COMM_WORLD);
				if (Loops == 1)
				{
					count[0] = PTSTRAStartStop[1] - PTSTRAStartStop[0];
					MPI_Send(&count[0], 1, MPI_INT, 1, 100010000 + myid * 1000000 + num*100, MPI_COMM_WORLD);//序号+来源*2+目的*2+等值*2+循环*2
					if (count[0] != 0)
					{
						MPI_Send(&PTS[0][PTSTRAStartStop[0] * 3], count[0] * 3, MPI_FLOAT, 1, 200010000 + myid * 1000000 + num*100, MPI_COMM_WORLD);

						MPI_Send(&PTSTRAStartStop[0], 1, MPI_INT, 1, 300010000 + myid * 1000000 + num * 100, MPI_COMM_WORLD);
					}

					count[1] = PTSTRAStartStop[3] - PTSTRAStartStop[2];
					MPI_Send(&count[1], 1, MPI_INT, 1, 400010000 + myid * 1000000 + num * 100, MPI_COMM_WORLD);
					if (count[1] != 0)
						MPI_Send(&TRA[0][PTSTRAStartStop[2] * 3], count[1] * 3, MPI_INT, 1, 500010000 + myid * 1000000 + num * 100, MPI_COMM_WORLD);
				}
				else
				{
					for (int L = 0; L < Loops; L++)
					{
						if (L == 0)
						{
							count[L * 2] -= PTSTRAStartStop[0];
							MPI_Send(&count[L * 2], 1, MPI_INT, 1, 100010000 + myid * 1000000 + num * 100+L, MPI_COMM_WORLD);
							if (count[L * 2] != 0)
							{
								MPI_Send(&PTS[L][PTSTRAStartStop[0] * 3], count[L * 2] * 3, MPI_FLOAT, 1, 200010000 + myid * 1000000 + num * 100 + L, MPI_COMM_WORLD);

								MPI_Send(&PTSTRAStartStop[0], 1, MPI_INT, 1, 300010000 + myid * 1000000 + num * 100 + L, MPI_COMM_WORLD);
							}

							count[L * 2 + 1] -= PTSTRAStartStop[2];
							MPI_Send(&count[L * 2 + 1], 1, MPI_INT, 1, 400010000 + myid * 1000000 + num * 100 + L, MPI_COMM_WORLD);
							if (count[L * 2 + 1] != 0)
								MPI_Send(&TRA[L][PTSTRAStartStop[2] * 3], count[L * 2 + 1] * 3, MPI_INT, 1, 500010000 + myid * 1000000 + num * 100 + L, MPI_COMM_WORLD);

							continue;
						}
						else if (L == Loops - 1)
						{
							MPI_Send(&PTSTRAStartStop[1], 1, MPI_INT, 1, 100010000 + myid * 1000000 + num * 100 + L, MPI_COMM_WORLD);
							if (PTSTRAStartStop[1] != 0)
								MPI_Send(PTS[L], PTSTRAStartStop[1] * 3, MPI_FLOAT, 1, 200010000 + myid * 1000000 + num * 100 + L, MPI_COMM_WORLD);

							MPI_Send(&PTSTRAStartStop[3], 1, MPI_INT, 1, 400010000 + myid * 1000000 + num * 100 + L, MPI_COMM_WORLD);
							if (PTSTRAStartStop[3] != 0)
								MPI_Send(TRA[L], PTSTRAStartStop[3] * 3, MPI_INT, 1, 500010000 + myid * 1000000 + num * 100 + L, MPI_COMM_WORLD);

							continue;
						}
						else
						{
							MPI_Send(&count[L * 2], 1, MPI_INT, 1, 100010000 + myid * 1000000 + num * 100 + L, MPI_COMM_WORLD);
							if (count[L * 2] != 0)
								MPI_Send(PTS[L], count[L * 2] * 3, MPI_FLOAT, 1, 200010000 + myid * 1000000 + num * 100 + L, MPI_COMM_WORLD);

							MPI_Send(&count[L * 2 + 1], 1, MPI_INT, 1, 400010000 + myid * 1000000 + num * 100 + L, MPI_COMM_WORLD);
							if (count[L * 2 + 1] != 0)
								MPI_Send(TRA[L], count[L * 2 + 1] * 3, MPI_INT, 1, 500010000 + myid * 1000000 + num * 100 + L, MPI_COMM_WORLD);

							continue;
						}
					}
				}

				time(&t[1]);
				Time[1+ 4* myid] += t[1] - t[0];
				t[0] = t[1];

				if (num == numValues - 1)
					MPI_Send(&Time[4 * myid], 4, MPI_INT, 1, 50001 + myid * 100, MPI_COMM_WORLD);
			}

			if (myid == 1)
			{
				for (int read_i = 0; read_i < readloop; read_i++)
				{
					int i = 2 + read_i;

					MPI_Recv(&t[0], 1, MPI_INT, i, 1000100 + i * 10000 + num, MPI_COMM_WORLD, &status);

					int Loops = 0;
					MPI_Recv(&Loops, 1, MPI_INT, i, 2000100 + i * 10000 + num, MPI_COMM_WORLD, &status);

					float** PTS = new float* [Loops];
					int** TRA = new int* [Loops];
					for (int L = 0; L < Loops; L++)
					{
						int size = 0;

						MPI_Recv(&size, 1, MPI_INT, i, 100010000 + i * 1000000 +  num*100+L, MPI_COMM_WORLD, &status);
						if (size != 0)
						{
							CountPTS.push_back(size);

							PTS[L] = new float[size * 3];
							MPI_Recv(PTS[L], size * 3, MPI_FLOAT, i, 200010000 + i * 1000000 + num * 100 + L, MPI_COMM_WORLD, &status);
							globalPTS.push_back(PTS[L]);

							if (L == 0)
							{
								MPI_Recv(&size, 1, MPI_INT, i, 300010000 + i * 1000000 + num * 100 + L, MPI_COMM_WORLD, &status);
								CountPTSStart.push_back(globalTRA.size());
								CountPTSStart.push_back(size);
							}
						}

						MPI_Recv(&size, 1, MPI_INT, i, 400010000 + i * 1000000 + num * 100 + L, MPI_COMM_WORLD, &status);
						if (size != 0)
						{
							CountTRA.push_back(size);

							TRA[L] = new int[size * 3];
							MPI_Recv(TRA[L], size * 3, MPI_INT, i, 500010000 + i * 1000000 + num * 100 + L, MPI_COMM_WORLD, &status);
							globalTRA.push_back(TRA[L]);
						}
					}
					time(&t[1]);
					Time[1+4 * myid] += t[1] - t[0];
					t[0] = t[1];
				}

				ofstream ofile;
				string outfilePath = infilePath + "_rupmc_" +to_string(numprocs)+"_" + to_string(int(values[0])) + "_" + to_string(int(values[numValues - 1])) + "_tra.txt";
				if (num == 0)
					ofile.open(outfilePath.c_str());
				else
					ofile.open(outfilePath.c_str(), ios::app);
				int base = 0;
				for (; TRA_i < globalTRA.size(); TRA_i++)
				{
					int TRAStart;
					if (TRA_i == CountPTSStart[PTSStart_i])
					{
						PTSStart_i++;
						TRAStart = sizePTS + base - CountPTSStart[PTSStart_i];
						PTSStart_i++;
					}

					for (int idx = 0; idx < CountTRA[TRA_i] * 3; idx += 3)
					{
						ofile << globalTRA[TRA_i][idx] + TRAStart << " ";
						ofile << globalTRA[TRA_i][idx + 1] + TRAStart << " ";
						ofile << globalTRA[TRA_i ][idx + 2] + TRAStart << " ";
						ofile << endl;
					}
					base += CountPTS[TRA_i];
					sizeTRA += CountTRA[TRA_i];

					delete[] globalTRA[TRA_i];
				}
				ofile.close();

				outfilePath = infilePath + "_rupmc_" + to_string(numprocs) + "_" + to_string(int(values[0])) + "_" + to_string(int(values[numValues - 1])) + "_pts.txt";
				if (num == 0)
					ofile.open(outfilePath.c_str());
				else
					ofile.open(outfilePath.c_str(), ios::app);
				for (; PTS_i < globalPTS.size(); PTS_i++)
				{
					for (int idx = 0; idx < CountPTS[PTS_i] * 3; idx += 3)
					{
						ofile << globalPTS[PTS_i][idx] << " ";
						ofile << globalPTS[PTS_i][idx + 1] << " ";
						ofile << globalPTS[PTS_i][idx + 2] << " ";
						ofile << endl;
					}
					sizePTS += CountPTS[PTS_i];

					delete[] globalPTS[PTS_i];
				}
				ofile.close();
				time(&t[1]);
				time(&T[1]);
				Time[3+ 4 * myid] += t[1] - t[0];

				if (num == numValues - 1)
				{
					string outfilePath = infilePath + "_rupmc_" + to_string(numprocs) + "_" + to_string(int(values[0])) + "_" + to_string(int(values[numValues - 1])) + "_time.txt";
					ofstream ofile(outfilePath.c_str());

					for (int i = 0; i < numprocs; i++)
					{
						if (i != 1)
							MPI_Recv(&Time[4 * i], 4, MPI_INT, i, 50001 + i * 100, MPI_COMM_WORLD, &status);
					}

					for (int i = 0; i < numprocs; i++)
					{
						ofile << i << ":" << endl;
						if (i == 0)
							ofile << "Total time: " << Time[4 * i] + Time[1+ 4 * i] << endl;
						else if (i == 1)
							ofile << "Total time: " << Time[3+ 4 * i] << endl;
						else
							ofile << "Total time: " << Time[2+ 4 * i] << endl;

						ofile << "Reading time: " << Time[0+4 * i] << endl;
						ofile << "Transfering time: " << Time[1 + 4 * i] << endl;
						ofile << "Computing time: " << Time[2+ 4 * i] << endl;
						ofile << "Writing time: " << Time[3+ 4 * i] << endl;

						if (i != 0)
						{
							Time[0] += Time[4 * i];
							Time[1] += Time[1 + 4 * i];
							Time[2] += Time[2 + 4 * i];
							Time[3] += Time[3 + 4 * i];
						}
					}
					ofile << endl;
					ofile << "Total time: " << T[1] - T[0] << endl;
					ofile << "Reading time: " << Time[0] << endl;
					ofile << "Transfering time: " << T[1] - T[0]-Time[0] - Time[2] - Time[3] << endl;
					ofile << "Computing time: " << Time[2] << endl;
					ofile << "Writing time: " << Time[3] << endl;
					ofile << "PTS Size: " << sizePTS << endl;
					ofile << "TRA Size: " << sizeTRA << endl;
					ofile.close();
				}
			}
		}
	}

	MPI_Finalize();

	system("pause");

	return 0;
}

