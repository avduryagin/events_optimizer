#pragma once
#include "pybind11\pybind11.h"
#include "pybind11\numpy.h"
#include "Python.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <list>
namespace py = pybind11;
class optimizer
{
	size_t ncell = 0, nrow = 0;
	size_t niter = 0,maxiter=100;
	const size_t ncol = 2;
	float tolerance = 1e-3f, dt = 1.f;
	float* weight = nullptr;
	float* weight_ = nullptr;
	py::array_t<float, py::array::c_style | py::array::forcecast>* data = nullptr;

	float* rest = nullptr;
	float var = 1.f, mean = 0.f;	
	size_t* cell_count = nullptr;
	size_t* cell_index = nullptr;
	//void print();
	void place(size_t, size_t, float);
	float loc_variance(float*) const;
	//void copy_weight(size_t) const;	
	
	float const func(size_t, size_t) const;	
	float const diff_(size_t,size_t, size_t, float, float) const;
	void replace(size_t, size_t);
	float const optim_variance(size_t, size_t*);
	float const get(size_t, size_t) const;
	size_t cell_ = NULL;
	size_t worker = NULL;
	float min_variance = 0.f;
	std::string log;
	void run(size_t);
	//void print_weight();
	
public:
	optimizer(size_t, float, float);
	~optimizer();
	void fit(py::array_t<float, py::array::c_style | py::array::forcecast>*, py::array_t<int, py::array::c_style | py::array::forcecast>& , py::array_t<float, py::array::c_style | py::array::forcecast>&, float,unsigned int,unsigned int);
	void optimize();
	void optimize_parallel(unsigned int);
	float variance = 1.f;
	unsigned int niter_ = 0;

};
