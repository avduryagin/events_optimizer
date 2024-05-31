#include "lib.h"
#include <thread>


int main()
{
	std::thread th;
	std::cout << "hardware_concurrency " << th.hardware_concurrency << std::endl;
	size_t ncell = 10, ngroup = 3;
	float tolerance = 1e-3, dt = 1.;

	//generalized_optimizer op(ncell, dt, tolerance, ngroup);
	
	
}