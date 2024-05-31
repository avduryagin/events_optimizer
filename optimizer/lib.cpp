#include "lib.h"
#include <math.h>
#include <fstream>
#include <string>
optimizer::optimizer(size_t ncell_,float dt_,float tolerance_)
{
	//инизиализация базовых переменных в конструкторе.
	//ncell_ - число ячеек в распределении.
	//dt_ - шаг распределения. Если dt=1, то ncell - число дней.
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

 void optimizer::fit(py::array_t<float, py::array::c_style | py::array::forcecast>* data_ptr, py::array_t<int, py::array::c_style | py::array::forcecast>& indices, py::array_t<float, py::array::c_style | py::array::forcecast>& weight, py::array_t<float, py::array::c_style | py::array::forcecast>* mean_array, unsigned int njobs = 1, unsigned int maxiter = 100)
{
	//Инициализируем переменные. Входные данные:
	//data_ptr - 2D массив. data_ptr[:,0] - duration; data_ptr[:,1]- debit.
	//indices- 1D массив размером 1x nrow (nrow - число мероприятий). В нем каждому мероприятию сопоставляется индекс из массива weight, 
	//weight - 1D массив размером 1x ncell (ncell - число дней) - массив, в который пишутся значения дебита распределенных мероприятий. 
	//mean_array - 1D массив размерности 1x ncell (ncell - число дней) - массив, в котором задаются целевые значения недоборов для каждой cell. К ним стремятся значения weight.	
	//maxiter - ограничение на число итераций алгоритма оптимизации. Предельное число прогона всего списка мероприятий.
	//njobs - число ядер, на которых можно производлить расчет. Экспериментальная опция. Из-за врпемени на диспетчеризацию быстрее всего работает на одном ядре.
	this->nrow = data_ptr->shape(0);
	this->mean = 0;	
	this->cell_index = new size_t[this->nrow];
	this->cell_count = new size_t[this->nrow];
	this->rest = new float[this->nrow];
	this->penalty = new float[this->nrow];
	this->data = data_ptr;
	this->mean_array = mean_array;
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
	i = 0;
	while (i<this->ncell)
	{
		this->weight[i] -= this->mean_array->at(i);
		++i;
	}
	this->var = this->loc_variance(this->weight);
	this->variance = this->var;
	//Запускаем алгоритм оптимизации
	if (njobs > 1) { this->optimize_parallel(njobs); }
	else 
	{
		this->optimize();
	}
	//записывает результаты в выходные массивы. Нужно для Python.
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
	/*std::string path = "C:\\Users\\avduryagin\\source\\repos\\pycpp\\log_file1.txt";
	std::ofstream out;
	out.open(path);*/

	//Последовательно перебираем все мероприятия в поиске оптимального положения по метрике.
	//Если такое положение найдено, мероприятие перещается в него (cell).
	//Алгоритм останавливается:
	// 1. Если перебрав все мероприятия, ни одно из них не может сменить своё положение, уменьшив целевую функцию (Атрибут go=false).
	//2. Достигнут минимум целевой функции !(this->var > this->tolerance).
	//3. Достигнуто предельное число итераций, ограниченное атрибутом maxiter.
	
	while (go && (this->niter < this->maxiter) && (this->var > this->tolerance))
	{
		
		go = false;
		i = 0;
		++this->niter;
		while(i<this->nrow)
		{
			
			cell = this->ncell;
			cell_ = this->cell_index[i];
			//для мероприятия i вычисляется минимум значения целевой функции. Если мимимум достигается в ячейке отличной от текущей
			//- перемещаем мероприятие в новую ячейку.
			var = this->optim_variance(i, &cell);
			//out << this->log;
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
generalized_optimizer::generalized_optimizer(size_t ncell_, float dt_, float tolerance_, size_t ngroup_):optimizer{ncell_,dt_,tolerance_}
{
	
	this->ngroup = ngroup_;
	this->groups = new float[this->ngroup];
	this->groups_ = new float[this->ngroup];
	for (size_t i = 0; i < this->ngroup; i++) 
	{
		this->groups[i] = 0;
		this->groups_[i] = 0;
	
	}
};
generalized_optimizer::~generalized_optimizer() 
{
	delete[] this->groups_;
	delete[] this->groups;
};
void generalized_optimizer::fit(py::array_t<float, py::array::c_style | py::array::forcecast>* data_ptr,
	py::array_t<int, py::array::c_style | py::array::forcecast>& indices,
	py::array_t<float, py::array::c_style | py::array::forcecast>& weight,
	py::array_t<float, py::array::c_style | py::array::forcecast>& group,
	py::array_t<float, py::array::c_style | py::array::forcecast>* mean_array, 
	py::array_t<int, py::array::c_style | py::array::forcecast>* group_index_,
	unsigned int maxiter = 100)
{	/*std::string path = "C:\\Users\\avduryagin\\source\\repos\\pycpp\\log_file.txt";
	std::ofstream out;
	out.open(path);*/

	// 
	//Инициализируем переменные. Входные данные:
	//data_ptr - 2D массив. data_ptr[:,0] - duration; data_ptr[:,1]- debit.
	//indices- 1D массив размером 1x nrow (nrow - число мероприятий). В нем каждому мероприятию сопоставляется индекс из массива weight, 
	//weight - 1D массив размером 1x ncell (ncell - число дней) - массив, в который пишутся значения дебита распределенных мероприятий. 
	//group -1D массив размером 1x ngroup (ngroup - число групп) - массив, в который пишутся значения дебита распределенных групп мероприятий. 
	//mean_array - 1D массив размерности 1x ngroup (ngroup - число групп) - массив, в котором задаются целевые значения недоборов для каждой группы. К ним стремятся значения groups.	
	//group_index_ - 1D массив размерности 1x ncell (ncell - число дней) - массив, в котором задается индекс принадлежности к группе для каждой ячейки. Ячейки объединяются индексами в непрерывные группы.
	//  Уникальное число групп равно ngroup.
	//maxiter - ограничение на число итераций алгоритма оптимизации. Предельное число прогона всего списка мероприятий.
	
	this->nrow = data_ptr->shape(0);
	this->mean = 0;
	this->cell_index = new size_t[this->nrow];
	this->cell_count = new size_t[this->nrow];
	this->rest = new float[this->nrow];
	this->penalty = new float[this->nrow];
	this->data = data_ptr;
	this->mean_array = mean_array;
	this->group_index = group_index_;
	size_t i = 0, size = 0;
	float widht, hight, rest, s = 0;
	this->maxiter = maxiter;


	while (i < this->nrow)
	{
		widht = this->data->at(i, 0);
		hight = this->data->at(i, 1);
		s = widht / this->dt;
		size = (size_t)floorf(s);
		this->cell_count[i] = size;
		rest = (s - size) * hight;
		this->rest[i] = rest;
		this->penalty[i] = this->penalty_(i, 0);
		//this->cell_index->mutable_at(i) = 0;
		this->cell_index[i] = 0;
		this->place(i, 0, 1.f);
		++i;
	}
	
	i = 0;
	while (i < this->ngroup)	{
		
		this->groups[i] -= this->mean_array->at(i);		
		++i;
	}
	
	this->var = this->loc_variance(this->groups);	
	this->variance = this->var;	
	//Запускаем алгоритм оптимизации
	this->optimize();
	
	//записывает результаты в выходные массивы. Нужно для Python.
	int j = 0;
	while (j < this->nrow)
	{
		indices.mutable_at(j) = (int)this->cell_index[j];
		//indices.mutable_at(j) = j;
		++j;
	}
	j = 0;
	while (j < this->ncell)
	{
		weight.mutable_at(j) = this->weight[j];
		++j;
	}
	j = 0;
	while (j < this->ngroup)
	{
		group.mutable_at(j) = this->groups[j];
		++j;
	}
	//out.close();
}
float generalized_optimizer::loc_variance(float*) const 
{
	size_t i = 0;
	float s = 0,s_=0;
	while(i<this->ngroup)
	{
		s_ = this->groups[i];
		s += (s_ * s_);
		++i;
	}
	return s / this->ngroup;
};
void generalized_optimizer::place(size_t index, size_t cell, float sign = -1.f)
{
	if (!(index < this->nrow) || !(cell < this->ncell)) { return; }
	size_t n, ncell;
	int teta;
	float val = 0,re=0;
	n = this->cell_count[index];
	val = get(index, 1) * sign;

	if (cell + n < this->ncell) 
	{ 
		re = this->rest[index] * sign;
		this->weight[cell + n] += re;		
		teta = this->group_index->at(cell + n);
		this->groups[teta] += re;	

	}

	ncell = std::min(cell + n, this->ncell);

	while (cell < ncell)
	{
		this->weight[cell] += val;
		teta = this->group_index->at(cell);
		this->groups[teta] += val;
		++cell;
	}

};

float generalized_optimizer::move_left(size_t index,float f0, float fn, float df, size_t n, size_t start, size_t cell, size_t& newcell,float dv) 
{
	size_t i = 0,cell_=0,tail=0,teta0=0,teta1=0,teta2=0;
	float v0 = 0, v1 = 0, v2 = 0, fteta0 = 0, fteta1 = 0, fteta2 = 0,f02,fn2,df2,dv_,dvmin,penalty;
	f02 = f0 * f0;
	fn2 = fn * fn;
	df2 = df * df;
	dvmin = 0;
	penalty = this->penalty[index];
	i = 0;
	while(i<start)
	{
		this->groups_[i] = this->groups[i];
		++i;
	}
	cell_ = cell;
	while(cell>0)
	{
		tail = cell + n;
		v0 = 0, v1 = 0, v2 = 0;
		teta2 = this->group_index->at(cell-1);
		fteta2 = this->groups_[teta2];
		v2 = 2 * fteta2 * f0 + f02;
		fteta2 += f0;
		this->groups_[teta2] += f0;

		if (tail>this->ncell)
		{
			dv_ = (v0 + v1 + v2) / this->ngroup;
			dv += dv_;
			cell -= 1;
			if ((dv<dvmin)&&(abs(dv-dvmin)>this->tolerance))
			{
				dvmin = dv;
				cell_ = cell;			
			}
			continue;
		}
		teta0 = this->group_index->at(tail - 1);
		fteta0 = this->groups_[teta0];
		if (teta0 == teta2) { fteta0 = fteta2; }
		v0 = 2 * fteta0 * df + df2;
		fteta0 += df;
		this->groups_[teta0] += df;

		if (tail == this->ncell)
		{
			dv_ = (v0 + v1 + v2) / this->ngroup;
			dv += dv_;
			cell -= 1;
			if ((dv < dvmin) && (abs(dv - dvmin) > this->tolerance))
			{
				dvmin = dv;
				cell_ = cell;
			}
			continue;
		}

		teta1 = this->group_index->at(tail);
		fteta1 = this->groups_[teta1];
		v1 = -2 * fteta1 * fn + fn2;
		fteta1 -= fn;
		this->groups_[teta1] -= fn;
		dv_ = (v0 + v1 + v2) / this->ngroup;
		dv += dv_;
		cell -= 1;
		if ((dv < dvmin) && (abs(dv - dvmin) > this->tolerance))
		{
			dvmin = dv;
			cell_ = cell;
		}

	
	}
	newcell = cell_;
	return dvmin;
};
float generalized_optimizer::move_right(size_t index, float f0, float fn, float df, size_t n, size_t start, size_t cell, size_t& newcell, float dv)
{
	size_t i = 0, cell_ = 0, tail = 0, teta0 = 0, teta1 = 0, teta2 = 0;
	float v0 = 0, v1 = 0, v2 = 0, fteta0 = 0, fteta1 = 0, fteta2 = 0, f02, fn2, df2, dv_, dvmin, penalty;
	f02 = f0 * f0;
	fn2 = fn * fn;
	df2 = df * df;
	dvmin = 0;
	penalty = this->penalty[index];
	i = start;
	while (i < this->ngroup)
	{
		this->groups_[i] = this->groups[i];		
		++i;
	}
	
	cell_ = cell;
	while(cell<this->ncell-1)
	{
		tail = cell + n+1;
		v0 = 0, v1 = 0, v2 = 0;
		teta0 = this->group_index->at(cell);
		fteta0 = this->groups_[teta0];
		v0 = -2 * fteta0 * f0 + f02;
		fteta0 -= f0;
		this->groups_[teta0] -= f0;		
		if (tail > this->ncell)
		{
			dv_ = (v0 + v1 + v2) / this->ngroup;
			dv += dv_;
			cell += 1;
			if ((dv < dvmin) && (abs(dv - dvmin) > this->tolerance))
			{
				dvmin = dv;
				cell_ = cell;
				
			}
			
			continue;
		}

		teta1 = this->group_index->at(tail-1);
		fteta1 = this->groups_[teta1];
		if (teta1 == teta0) { fteta1 = fteta0; }
		v1 = -2 * fteta1 * df + df2;
		fteta1 -= df;
		this->groups_[teta1] -= df;

		if (tail == this->ncell)
		{
			dv_ = (v0 + v1 + v2) / this->ngroup;
			dv += dv_;
			cell += 1;
			if ((dv < dvmin) && (abs(dv - dvmin) > this->tolerance))
			{
				dvmin = dv;
				cell_ = cell;
				
			}
			continue;
		}
		teta2 = this->group_index->at(tail);
		fteta2 = this->groups_[teta2];	
		if (teta2 == teta1) { fteta2 = fteta1; }
		v2 = 2 * fteta2 * fn + fn2;
		fteta2 += fn;
		this->groups_[teta2] += fn;

		dv_ = (v0 + v1 + v2) / this->ngroup;
		dv += dv_;
		cell += 1;
		if ((dv < dvmin) && (abs(dv - dvmin) > this->tolerance))
		{
			dvmin = dv;
			cell_ = cell;
		}

	}
	newcell = cell_;
	return dvmin;
};
float const generalized_optimizer::optim_variance(size_t index, size_t* newcell) 
{
	size_t current_cell, size, rstart, lstart, lstart_, rcell, lcell;
	float f0, fn, df,rmin,lmin,vmin=0;
	current_cell = this->cell_index[index];
	size = this->cell_count[index];
	f0 = this->get(index, 1);
	fn = this->rest[index];
	df = fn - f0;
	rstart = this->group_index->at(current_cell);
	lstart_ = std::min(current_cell + size, this->ncell - 1);
	lstart = this->group_index->at(lstart_)+1;
	rmin = this->move_right(index,f0, fn, df, size, rstart, current_cell, rcell,0.f);
	lmin = this->move_left(index, f0, fn, df, size, lstart, current_cell, lcell, 0.f);
	if (rmin < lmin) { *newcell = rcell; vmin = rmin; }
	else { *newcell = lcell; vmin = lmin; };
	return vmin; 

};