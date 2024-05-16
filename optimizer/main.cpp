#include "lib.h"
#include <thread>


int main()
{
	std::thread th;
	std::cout << "hardware_concurrency " << th.hardware_concurrency << std::endl;
	
}