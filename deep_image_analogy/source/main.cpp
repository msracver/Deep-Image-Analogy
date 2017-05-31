#include "DeepAnalogy.cuh"

int main(int argc, char** argv) {

	DeepAnalogy dp;

	if (argc!=9) {

		string model = "models/";
	
		string A = "demo/content.png";
		string BP = "demo/style.png";
		string output = "demo/output/";

		dp.SetModel(model);
		dp.SetA(A);
		dp.SetBPrime(BP);
		dp.SetOutputDir(output);
		dp.SetGPU(0);
		dp.SetRatio(0.5);
		dp.SetBlendWeight(2);
		dp.UsePhotoTransfer(false);
		dp.LoadInputs();
		dp.ComputeAnn();
		
	}
	else{
		dp.SetModel(argv[1]);
		dp.SetA(argv[2]);
		dp.SetBPrime(argv[3]);
		dp.SetOutputDir(argv[4]);
		dp.SetGPU(atoi(argv[5]));
		dp.SetRatio(atof(argv[6]));
		dp.SetBlendWeight(atoi(argv[7]));
		if (atoi(argv[8]) == 1) {
			dp.UsePhotoTransfer(true);
		}
		else{
			dp.UsePhotoTransfer(false);
		}
		dp.LoadInputs();
		dp.ComputeAnn();
	}



	return 0;
}
