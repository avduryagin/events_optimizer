#include "lib.h"
#include <math.h>
#include <fstream>
#include <string>
optimizer::optimizer(size_t ncell_,float dt_,float tolerance_)
{
	//инизиализаци€ базовых переменных в конструкторе.
	//ncell_ - число €чеек в распределении.
	//dt_ - шаг распределени€. ≈сли dt=1, то ncell - число дней.
	//tolerance_ - точность алгоритма. 
	this->ncell = ncell_;
	this->penalty_cell = this->ncell - 1;
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
	if (this->penalty != 0)
	{
		delete[] this->penalty;
	}
}

void optimizer::fit(py::array_t<float, py::array::c_style | py::array::forcecast>* data_ptr, py::array_t<int, py::array::c_style | py::array::forcecast>& indices, py::array_t<float, py::array::c_style | py::array::forcecast>& weight, float mean_ = 0., unsigned int njobs = 1, unsigned int maxiter = 100)
{
	//»нициализируем переменные. ¬ходные данные:
	//data_ptr - 2D массив. data_ptr[:,0] - duration; data_ptr[:,1]- debit.
	//indices- 1D массив размером 1x nrow (nrow - число меропри€тий). ¬ нем каждому меропри€тию сопоставл€етс€ индекс из массива weight, 
	//weight - 1D массив размером 1x ncell (ncell - число дней) - массив, в который пишутс€ значени€ дебита распределенных меропри€тий. 
	//mean - среднее значение, к которому стрем€тс€ значени€ weight.
	//maxiter - ограничение на число итераций алгоритма оптимизации. ѕредельное число прогона всего списка меропри€тий.
	//njobs - число €дер, на которых можно производлить расчет. Ёкспериментальна€ опци€. »з-за врпемени на диспетчеризацию быстрее всего работает на одном €дре.
	this->nrow = data_ptr->shape(0);
	this->mean = mean_;	
	this->cell_index = new size_t[this->nrow];
	this->cell_count = new size_t[this->nrow];
	this->rest = new float[this->nrow];
	this->penalty = new float[this->nrow];
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
		this->penalty[i] = this->penalty_(i, 0);
		//this->cell_index->mutable_at(i) = 0;
		this->cell_index[i] = 0;
		this->place(i, 0,1.f);
		++i;
	}
	this->var = this->loc_variance(this->weight);
	this->variance = this->var;
	//«апускаем алгоритм оптимизации
	if (njobs > 1) { this->optimize_parallel(njobs); }
	else 
	{
		this->optimize();
	}
	//записывает результаты в выходные массивы. Ќужно дл€ Python.
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
	//std::string path = "C:\\Users\\avduryagin\\source\\repos\\pycpp\\log_file.txt";
	//std::ofstream out;
	//out.open(path);

	//ѕоследовательно перебираем все меропри€ти€ в поиске оптимального положени€ по метрике.
	//≈сли такое положение найдено, меропри€тие перещаетс€ в него (cell).
	//јлгоритм останавливаетс€:
	// 1. ≈сли перебрав все меропри€ти€, ни одно из них не может сменить своЄ положение, уменьшив целевую функцию (јтрибут go=false).
	//2. ƒостигнут минимум целевой функции !(this->var > this->tolerance).
	//3. ƒостигнуто предельное число итераций, ограниченное атрибутом maxiter.
	
	while (go && (this->niter < this->maxiter) && (this->var > this->tolerance))
	{
		
		go = false;
		i = 0;
		++this->niter;
		while(i<this->nrow)
		{
			
			cell = this->ncell;
			cell_ = this->cell_index[i];
			//дл€ меропри€ти€ i вычисл€етс€ минимум значени€ целевой функции. ≈сли мимимум достигаетс€ в €чейке отличной от текущей
			//- перемещаем меропри€тие в новую €чейку.
			var = this->optim_variance(i, &cell);
			if ((cell < this->ncell)&&(cell!=cell_))
			{
				
				this->replace(i, cell);
				this->penalty[i] = var;
				cell = this->ncell;
				go = true;
				
			}
			
			++ i;
		}
		
		
		this->var = this->loc_variance(this->weight);	
	}
	
	this->variance = this->var;
	this->niter_ = (unsigned int)this->niter;
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
	
	float min_var= this->penalty[index];
	float var=0.f,f0,f1,df=0,delta=0,penalty=0,penalty_=0;

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
		penalty = this->penalty_(index, i - 1);
		var -= df;
		delta = std::abs(var - min_var);
		if(((var+penalty)<min_var)&&(delta>this->tolerance))
		{ 
			min_var = var;
			penalty_ = penalty;
			cell = i-1;
			
		}	

		--i;
	}

	i = current_cell;
	var = 0;
	while (i<this->ncell-1)
	{
		df = this->diff_(index,i, size, f0, f1);
		penalty = this->penalty_(index, i+1);		

		var += df;
		delta = std::abs(var - min_var);
		if (((var+penalty) < min_var) && (delta > this->tolerance))
		{
			min_var = var;
			penalty_ = penalty;
			cell = i+1;
		}

		++i;

	
	}	
	if (cell!=current_cell){ *new_cell = cell; }
	else { *new_cell = this->ncell; }	
	return penalty_;

}

float const optimizer::penalty_(size_t index, size_t cell)
{
	int npenalty = cell + this->cell_count[index] - this->penalty_cell;
	float penalty = 0.f;
	
	
	if (!(npenalty > 0)) { return penalty; }
	float debit,tau;
	debit = this->get(index,1);
	tau= this->get(index, 0);
	penalty= debit * (tau - this->cell_count[index] - 1 + npenalty);
	//this->log.append("i=" + std::to_string(index) + ", cell=" + std::to_string(cell) + ", npenalty =" + std::to_string(npenalty)+ ", penalty =" + std::to_string(penalty)+"\n");
	return penalty;


};
