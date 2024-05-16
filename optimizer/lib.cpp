#include "lib.h"
#include <math.h>
#include <fstream>
#include <string>
optimizer::optimizer(size_t ncell_,float dt_,float tolerance_)
{
	//������������� ������� ���������� � ������������.
	//ncell_ - ����� ����� � �������������.
	//dt_ - ��� �������������. ���� dt=1, �� ncell - ����� ����.
	//tolerance_ - �������� ���������. 
	this->ncell = ncell_;
	if (dt_ > 0) {this->dt = dt_;}
	if (tolerance_ > 0) { this->tolerance = tolerance_; }
	this->weight = new float[this->ncell];
	this->weight_ = new float[this->ncell];
	size_t i = 0;
	while (i<this->ncell)
	{
		weight[i] = 0;
		weight_[i] = 0;
		++i;
	}
}
optimizer::~optimizer()
{
	delete[] this->weight;
	delete[] this->weight_;
	if (this->cell_count!=0)
	{
		delete[] this->cell_count;
	}
	if (this->cell_index != 0)
	{
		delete[] this->cell_index;
	}
	if (this->rest != 0)
	{
		delete[] this->rest;
	}
}

void optimizer::fit(py::array_t<float, py::array::c_style | py::array::forcecast>* data_ptr, py::array_t<int, py::array::c_style | py::array::forcecast>& indices, py::array_t<float, py::array::c_style | py::array::forcecast>& weight, float mean_ = 0., unsigned int njobs = 1, unsigned int maxiter = 100)
{
	//�������������� ����������. ������� ������:
	//data_ptr - 2D ������. data_ptr[:,0] - duration; data_ptr[:,1]- debit.
	//indices- 1D ������ �������� 1x nrow (nrow - ����� �����������). � ��� ������� ����������� �������������� ������ �� ������� weight, 
	//weight - 1D ������ �������� 1x ncell (ncell - ����� ����) - ������, � ������� ������� �������� ������ �������������� �����������. 
	//mean - ������� ��������, � �������� ��������� �������� weight.
	//maxiter - ����������� �� ����� �������� ��������� �����������. ���������� ����� ������� ����� ������ �����������.
	//njobs - ����� ����, �� ������� ����� ������������ ������. ����������������� �����. ��-�� �������� �� ��������������� ������� ����� �������� �� ����� ����.
	this->nrow = data_ptr->shape(0);
	this->mean = mean_;	
	this->cell_index = new size_t[this->nrow];
	this->cell_count = new size_t[this->nrow];
	this->rest = new float[this->nrow];
	this->data = data_ptr;
	size_t i = 0,size=0;
	float widht, hight,rest,s=0;
	this->maxiter = maxiter;
	
	while(i<this->nrow)
	{
		widht = this->data->at(i,0);
		hight= this->data->at(i,1);
		s = widht / this->dt;
		size=(size_t)floorf(s);
		this->cell_count[i] = size;
		rest = (s - size) * hight;
		this->rest[i] = rest;
		//this->cell_index->mutable_at(i) = 0;
		this->cell_index[i] = 0;
		this->place(i, 0,1.f);
		++i;
	}
	this->var = this->loc_variance(this->weight);
	this->variance = this->var;
	//��������� �������� �����������
	if (njobs > 1) { this->optimize_parallel(njobs); }
	else 
	{
		this->optimize();
	}
	//���������� ���������� � �������� �������. ����� ��� Python.
	int j = 0;
	while (j < this->nrow)
	{
		indices.mutable_at(j) = (int)this->cell_index[j];
		//indices.mutable_at(j) = j;
		++j;
	}
	j = 0;
	while(j<this->ncell)
	{
		weight.mutable_at(j) = this->weight[j];
		++j;
	}
}

void optimizer::optimize()
{
	//float var,var_,lvar;
	size_t cell, i, cell_;
	bool go = true;
	this->niter = 0;
	//std::string path = "D:\\log_file.txt";
	//std::ofstream out;
	

	//��������������� ���������� ��� ����������� � ������ ������������ ��������� �� �������.
	//���� ����� ��������� �������, ����������� ���������� � ���� (cell).
	//�������� ���������������:
	// 1. ���� �������� ��� �����������, �� ���� �� ��� �� ����� ������� ��� ���������, �������� ������� ������� (������� go=false).
	//2. ��������� ������� ������� ������� !(this->var > this->tolerance).
	//3. ���������� ���������� ����� ��������, ������������ ��������� maxiter.
	
	while (go && (this->niter < this->maxiter) && (this->var > this->tolerance))
	{
		
		go = false;
		i = 0;
		++this->niter;
		while(i<this->nrow)
		{
			
			cell = this->ncell;
			cell_ = this->cell_index[i];
			//��� ����������� i ����������� ������� �������� ������� �������. ���� ������� ����������� � ������ �������� �� �������
			//- ���������� ����������� � ����� ������.
			var = this->optim_variance(i, &cell);

			if ((cell < this->ncell)&&(cell!=cell_))
			{
				//this->log.append(std::to_string(i) + " " + std::to_string(var) + " " + std::to_string(cell) + "\n");
				this->replace(i, cell);
				cell = this->ncell;
				go = true;
				
			}
			
			++ i;
		}

		this->var = this->loc_variance(this->weight);	
	}
	
	this->variance = this->var;
	this->niter_ = (unsigned int)this->niter;
	//out.open(path);
	//out << this->log;
	//out.close();
}
void optimizer::run(size_t index)
{
	std::mutex m;
	float var;
	size_t cell = this->ncell;
	var = this->optim_variance(index, &cell);
	//delta = abs(var - this->min_variance);
	
	if ((cell<this->ncell) && (var < this->min_variance))
	{
		m.lock();
		this->min_variance = var;
		this->cell_ = cell;
		this->worker = index;
		m.unlock();	
	}
		

}
void optimizer::optimize_parallel(unsigned int njobs_)
{
	std::thread th;	
	size_t i,njobs,niter;
	njobs = std::min(njobs_, th.hardware_concurrency());
	bool go = true;
	std::queue<size_t> main_queue;
	std::list<size_t> indices;

	this->niter = 0;
	while (go && (this->niter<this->maxiter) && (this->var > this->tolerance))
	{


		i = 0;
		while (i < this->nrow) { main_queue.push(i); ++i; }

		go = false;	
		niter = 0;
		while (!main_queue.empty())
		{


			while ((indices.size() < njobs) && !(main_queue.empty()))
			{
				indices.push_back(main_queue.front());
				main_queue.pop();
			}

			this->cell_ = this->ncell;
			this->min_variance = 0;
			for (auto iter = indices.begin(); iter != indices.end(); iter++)
			{
				std::thread th_(&optimizer::run,this,*iter);
				th_.join();

			}

			++niter;


			if (this->cell_ <this->ncell)
			{
				indices.remove(this->worker);				
				this->replace(this->worker, this->cell_);				
				go = true;

			}
			else { indices.clear(); }
		}
		++this->niter;
		this->var = this->loc_variance(this->weight);

	}

	this->variance = this->var;
}



float optimizer::loc_variance( float* weight) const 
{
	float var = 0.f;
	size_t i = 0;
	while(i<this->ncell)
	{
		//var += std::powf(weight[i] - this->mean, 2);
		var += (weight[i] - this->mean) * (weight[i] - this->mean);
		++i;
	}
	return var / this->ncell;
}


float const optimizer::get(size_t i,size_t j) const 
{
	return this->data->at(i,j);
}

void optimizer::place(size_t index,size_t cell,float sign=-1.f)
{ if(!(index<this->nrow)||!(cell<this->ncell)){return;}
size_t n,ncell;
float val = 0;
n = this->cell_count[index];
val = get(index, 1) * sign;

if (cell+n<this->ncell){this->weight[cell + n]+=this->rest[index]*sign;}

ncell = std::min(cell + n, this->ncell);

while(cell<ncell)
{
	this->weight[cell] += val;	
	++cell;
}

}
void optimizer::replace(size_t index,size_t new_cell)
{
	if (!(index < this->nrow) || !(new_cell < this->ncell)) { return; }
	size_t current_cell = this->cell_index[index];
	this->place(index, current_cell, -1.f);
	this->place(index, new_cell, 1.f);
	this->cell_index[index] = new_cell;

}

float const optimizer::func(size_t index,size_t i) const
{
	if (i<this->ncell)
	{
		if ((i >= this->cell_index[index]) && (i < this->cell_index[index] + this->cell_count[index])) { return get(index, 1); }
		else if (this->cell_index[index] + this->cell_count[index] == i) { return this->rest[index]; }
		else return 0.f;
	}
	else { return 0.f; }
}

float const optimizer::diff_(size_t index,size_t cell, size_t size, float f0, float f1) const
{
	float wp = 0.f, wph_ = 0.f, wph = 0.f, val = 0;
	wp = this->weight[cell]- this->func(index, cell);
	if (cell + size < this->ncell)
	{
		wph_ = this->weight[cell + size - 1] - this->func(index, cell + size - 1);
		wph = this->weight[cell + size] - this->func(index, cell + size);
		val = 2 * (f0 * (wph_ - wp) + f1 * (wph - wph_)) / this->ncell;
		return val;
	}
	if (cell + size - 1 < this->ncell)
	{
		wph_ = this->weight[cell + size - 1]- this->func(index, cell + size - 1);
		val = (2 * (f0 * (wph_ - wp) - f1 * (wph_ - this->mean)) - f1 * f1) / this->ncell;
		return val;
	}
	val = (f0 * f0 + 2 * (wp - this->mean) * f0) * -1.f / this->ncell;
	return val;
}

float const optimizer::optim_variance(size_t index, size_t* new_cell)
{
	float min_var= 0;
	float var=0.f,f0,f1,df=0,delta=0;

	size_t current_cell = this->cell_index[index];
	size_t size= this->cell_count[index]+1;
	f0 = this->get(index, 1);
	f1 = this->rest[index];
	size_t cell=current_cell,i=0;
	i = current_cell;
	var = 0;

	while (i>0)
	{
		df = this->diff_(index,i-1, size, f0, f1);
		var -= df;
		delta = std::abs(var - min_var);
		if((var<min_var)&&(delta>this->tolerance))
		{ 
			min_var = var;
			cell = i-1;
			
		}	

		--i;
	}

	i = current_cell;
	var = 0;
	while (i<this->ncell-1)
	{
		df = this->diff_(index,i, size, f0, f1);
		var += df;
		delta = std::abs(var - min_var);
		if ((var < min_var) && (delta > this->tolerance))
		{
			min_var = var;
			cell = i+1;
		}

		++i;

	
	}

	if (cell!=current_cell){ *new_cell = cell; }
	else { *new_cell = this->ncell; }	
	return min_var;

}