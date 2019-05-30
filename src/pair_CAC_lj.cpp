/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "pair_CAC_lj.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "update.h"
#include "neigh_list.h"
#include "timer.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include "asa_user.h"

#define MAXNEIGH1  50
#define MAXNEIGH2  10
//#include "math_extra.h"


using namespace LAMMPS_NS;
using namespace MathConst;
using namespace std;


/* ---------------------------------------------------------------------- */

PairCACLJ::PairCACLJ(LAMMPS *lmp) : PairCAC(lmp)
{
  restartinfo = 0;
  nmax = 0;
  outer_neighflag = 0;

  interior_scales = NULL;
  surface_counts = NULL;
 

  inner_neighbor_coords = NULL;
 
  inner_neighbor_types = NULL;
  

  surface_counts_max[0] = 0;
  surface_counts_max[1] = 0;
  surface_counts_max[2] = 0;
  surface_counts_max_old[0] = 0;
  surface_counts_max_old[1] = 0;
  surface_counts_max_old[2] = 0;
}

/* ---------------------------------------------------------------------- */

PairCACLJ::~PairCACLJ() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
    memory->destroy(mass_matrix);
    //memory->destroy(force_density_interior);
	memory->destroy(inner_neighbor_coords);
	
	memory->destroy(inner_neighbor_types);


  }
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairCACLJ::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  max_nodes_per_element = atom->nodes_per_element;
  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");


  memory->create(mass_matrix, max_nodes_per_element, max_nodes_per_element,"pairCAC:mass_matrix");
  memory->create(mass_copy, max_nodes_per_element, max_nodes_per_element,"pairCAC:copy_mass_matrix");
  memory->create(force_column, max_nodes_per_element,3,"pairCAC:force_residue");
  memory->create(current_force_column, max_nodes_per_element,"pairCAC:current_force_residue");
  memory->create(current_nodal_forces, max_nodes_per_element,"pairCAC:current_nodal_force");
  memory->create(pivot, max_nodes_per_element+1,"pairCAC:pivots");
  memory->create(surf_set, 6, 2, "pairCAC:surf_set");
  memory->create(dof_set, 6, 4, "pairCAC:surf_set");
  memory->create(sort_surf_set, 6, 2, "pairCAC:surf_set");
  memory->create(sort_dof_set, 6, 4, "pairCAC:surf_set");
  quadrature_init(2);
}
/* ----------------------------------------------------------------------
global settings
------------------------------------------------------------------------- */
void PairCACLJ::settings(int narg, char **arg) {
	if (narg <1 || narg>2) error->all(FLERR, "Illegal pair_style command");

	//cutmax = force->numeric(FLERR, arg[0]);
	

	//cut_global_s = force->numeric(FLERR,arg[1]);
	//neighrefresh = force->numeric(FLERR, arg[1]);
	//maxneigh_setting = force->numeric(FLERR, arg[2]);
	
	force->newton_pair = 0;
	cut_global_s = force->numeric(FLERR, arg[0]);
	if (narg == 2) {
		if (strcmp(arg[1], "one") == 0) atom->one_layer_flag=one_layer_flag = 1;
		else error->all(FLERR, "Unexpected argument in pair style CAC/lj invocation; only accepts cutoff and the 'one' keyword");
	}

	if (allocated) {
		int i, j;
		for (i = 1; i <= atom->ntypes; i++)
			for (j = i; j <= atom->ntypes; j++)
				if (setflag[i][j]) cut[i][j] = cut_global_s;
	}
	// reset cutoffs that have been explicitly set
	// initialize unit cell vectors

	
}


/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairCACLJ::coeff(int narg, char **arg) {
  if (narg < 4 || narg > 5)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR, arg[0], atom->ntypes, ilo, ihi);
  force->bounds(FLERR, arg[1], atom->ntypes, jlo, jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);

  double cut_one = cut_global_s;
  if (narg == 5) cut_one = force->numeric(FLERR,arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairCACLJ::init_one(int i, int j) {

if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag) {
    double ratio = sigma[i][j] / cut[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // check interior rRESPA cutoff

  //if (cut_respa && cut[i][j] < cut_respa[3])
    //error->all(FLERR,"Pair cutoff < Respa interior cutoff");

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);

    double sig2 = sigma[i][j]*sigma[i][j];
    double sig6 = sig2*sig2*sig2;
    double rc3 = cut[i][j]*cut[i][j]*cut[i][j];
    double rc6 = rc3*rc3;
    double rc9 = rc3*rc6;
    etail_ij = 8.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (sig6 - 3.0*rc6) / (9.0*rc9);
    ptail_ij = 16.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (2.0*sig6 - 3.0*rc6) / (9.0*rc9);
  }
 	
	return cut_global_s;
}

/* ---------------------------------------------------------------------- */


void PairCACLJ::init_style()
{
  check_existence_flags();
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style CAC_LJ requires atom IDs");

  maxneigh_quad_inner = MAXNEIGH2;
  maxneigh_quad_outer = MAXNEIGH1;
  // need a full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  //neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->CAC = 1;
  //surface selection array 
  surf_set[0][0] = 1;
  surf_set[0][1] = -1;
  surf_set[1][0] = 1;
  surf_set[1][1] = 1;
  surf_set[2][0] = 2;
  surf_set[2][1] = -1;
  surf_set[3][0] = 2;
  surf_set[3][1] = 1;
  surf_set[4][0] = 3;
  surf_set[4][1] = -1;
  surf_set[5][0] = 3;
  surf_set[5][1] = 1;

  //surface DOF array

  dof_set[0][0] = 0;
  dof_set[0][1] = 3;
  dof_set[0][2] = 4;
  dof_set[0][3] = 7;

  dof_set[1][0] = 1;
  dof_set[1][1] = 2;
  dof_set[1][2] = 5;
  dof_set[1][3] = 6;

  dof_set[2][0] = 0;
  dof_set[2][1] = 1;
  dof_set[2][2] = 4;
  dof_set[2][3] = 5;

  dof_set[3][0] = 2;
  dof_set[3][1] = 3;
  dof_set[3][2] = 6;
  dof_set[3][3] = 7;

  dof_set[4][0] = 0;
  dof_set[4][1] = 1;
  dof_set[4][2] = 2;
  dof_set[4][3] = 3;

  dof_set[5][0] = 4;
  dof_set[5][1] = 5;
  dof_set[5][2] = 6;
  dof_set[5][3] = 7;

  for (int si = 0; si < 6; si++) {
	  sort_dof_set[si][0] = dof_set[si][0];
	  sort_dof_set[si][1] = dof_set[si][1];
	  sort_dof_set[si][2] = dof_set[si][2];
	  sort_dof_set[si][3] = dof_set[si][3];
	  sort_surf_set[si][0] = surf_set[si][0];
	  sort_surf_set[si][1] = surf_set[si][1];
  }

  //minimization algorithm parameters
  //asacg_parm scgParm;
  //asa_parm sasaParm;

  memory->create(cgParm, 1, "pairCAC:cgParm");

  memory->create(asaParm, 1, "pairCAC:asaParm");
  memory->create(Objective, 1, "pairCAC:asaParm");
  // if you want to change parameter value, initialize strucs with default 
  asa_cg_default(cgParm);
  asa_default(asaParm);

  // if you want to change parameters, change them here: 
  cgParm->PrintParms = FALSE;
  cgParm->PrintLevel = 0;

  asaParm->PrintParms = FALSE;
  asaParm->PrintLevel = 0;
  asaParm->PrintFinal = 0;


}







//-----------------------------------------------------------------------


void PairCACLJ::force_densities(int iii, double s, double t, double w, double coefficients,
	double &force_densityx, double &force_densityy, double &force_densityz) {

int internal;

double delx,dely,delz;

double r2inv;
double r6inv;
double shape_func;


int neighborflag=0;
int outofbounds=0;
int timestep=update->ntimestep;
double unit_cell_mapped[3];
double scanning_unit_cell[3];
double ****nodal_gradients=atom->nodal_gradients;
double *special_lj = force->special_lj;
double forcelj,factor_lj,fpair;
int *type = atom->type;
double unit_cell[3];
double distancesq;
double current_position[3];
double scan_position[3];
double rcut;

int nodes_per_element;
int *nodes_count_list = atom->nodes_per_element_list;	


int flagm;






//equivalent isoparametric cutoff range for a cube of rcut


unit_cell_mapped[0] = 2 / double(current_element_scale[0]);
unit_cell_mapped[1] = 2 / double(current_element_scale[1]);
unit_cell_mapped[2] = 2 / double(current_element_scale[2]);






unit_cell[0] = s;
unit_cell[1] = t;
unit_cell[2] = w;





//scan the surrounding unit cell locations in a cartesian grid
//of isoparametric space until the cutoff is exceeded
//for each grid scan


 scanning_unit_cell[0]=unit_cell[0];
 scanning_unit_cell[1]=unit_cell[1];
 scanning_unit_cell[2]=unit_cell[2];


int distanceflag=0;
    current_position[0]=0;
    current_position[1]=0;
    current_position[2]=0;

	if (!atomic_flag) {
		nodes_per_element = nodes_count_list[current_element_type];
		for (int kkk = 0; kkk < nodes_per_element; kkk++) {
			shape_func = shape_function(unit_cell[0], unit_cell[1], unit_cell[2], 2, kkk + 1);
			current_position[0] += current_nodal_positions[kkk][poly_counter][0] * shape_func;
			current_position[1] += current_nodal_positions[kkk][poly_counter][1] * shape_func;
			current_position[2] += current_nodal_positions[kkk][poly_counter][2] * shape_func;
		}
	}
	else {
		current_position[0] = s;
		current_position[1] = t;
		current_position[2] = w;
	}


	rcut = cut_global_s;
	int origin_type = type_array[poly_counter];



	//precompute virtual neighbor atom locations

	


			int listtype;
			int listindex;
			int poly_index;
			double force_contribution[3];
			int scan_type;
			int poly_grad_scan;
			int element_index;
			int *ilist, *jlist, *numneigh, **firstneigh;
			int neigh_max = inner_quad_lists_counts[iii][neigh_quad_counter];
			int **node_types = atom->node_types;
			ilist = list->ilist;
			numneigh = list->numneigh;
			firstneigh = list->firstneigh;
			jlist = firstneigh[iii];
			double ****nodal_positions = atom->nodal_positions;
				//if(update->ntimestep==1)
      
      if(neigh_max>local_inner_max){
			memory->grow(inner_neighbor_coords, neigh_max, 3, "Pair_CAC_lj:inner_neighbor_coords");

			memory->grow(inner_neighbor_types, neigh_max, "Pair_CAC_lj:inner_neighbor_types");
	     local_inner_max=neigh_max;
	     }
      
			for (int l = 0; l < neigh_max; l++) {
				scanning_unit_cell[0] = inner_quad_lists_ucell[iii][neigh_quad_counter][l][0];
		    scanning_unit_cell[1] = inner_quad_lists_ucell[iii][neigh_quad_counter][l][1];
		    scanning_unit_cell[2] = inner_quad_lists_ucell[iii][neigh_quad_counter][l][2];
		     //listtype = quad_list_container[iii].inner_list2ucell[neigh_quad_counter].cell_indexes[l][0];
		     listindex = inner_quad_lists_index[iii][neigh_quad_counter][l][0];
		    poly_index = inner_quad_lists_index[iii][neigh_quad_counter][l][1];
		    element_index = listindex;
		    element_index &= NEIGHMASK;
		    inner_neighbor_types[l] = node_types[element_index][poly_index];
		    neigh_list_cord(inner_neighbor_coords[l][0], inner_neighbor_coords[l][1], inner_neighbor_coords[l][2],
			  element_index, poly_index, scanning_unit_cell[0], scanning_unit_cell[1], scanning_unit_cell[2]);

			}
			
		  
			for (int l = 0; l < neigh_max; l++) {

				scan_type = inner_neighbor_types[l];
				scan_position[0] = inner_neighbor_coords[l][0];
				scan_position[1] = inner_neighbor_coords[l][1];
				scan_position[2] = inner_neighbor_coords[l][2];
				delx = current_position[0] - scan_position[0];
				dely = current_position[1] - scan_position[1];
				delz = current_position[2] - scan_position[2];
				distancesq = delx*delx + dely*dely + delz*delz;

				r2inv = 1.0 / distancesq;
				r6inv = r2inv*r2inv*r2inv;
				factor_lj = special_lj[sbmask(iii)];
				forcelj = r6inv * (lj1[scan_type][origin_type]
					* r6inv - lj2[scan_type][origin_type]);
				fpair = factor_lj*forcelj*r2inv;

        //timer->stamp(Timer::CAC_INIT);
				force_densityx += delx*fpair;
				force_densityy += dely*fpair;
				force_densityz += delz*fpair;
				force_contribution[0] = delx*fpair;
				force_contribution[1] = dely*fpair;
				force_contribution[2] = delz*fpair;
				//timer->stamp(Timer::CAC_FD);
				//if (0) {
        if (quad_eflag) {

				scanning_unit_cell[0] = inner_quad_lists_ucell[iii][neigh_quad_counter][l][0];
			  scanning_unit_cell[1] = inner_quad_lists_ucell[iii][neigh_quad_counter][l][1];
			  scanning_unit_cell[2] = inner_quad_lists_ucell[iii][neigh_quad_counter][l][2];
			  listindex = inner_quad_lists_index[iii][neigh_quad_counter][l][0];
			  poly_grad_scan = inner_quad_lists_index[iii][neigh_quad_counter][l][1];


					quadrature_energy += r6inv*(lj3[origin_type][scan_type] * r6inv - lj4[origin_type][scan_type])/2 -
						offset[origin_type][scan_type]/2;

					//evdwl *= factor_lj;
					//compute gradients of the potential with respect to nodes
					//the calculation is the chain rule derivative evaluation of the integral of the energy density
					//added for every internal degree of freedom i.e. u=u1+u2+u3..upoly defines the total unit cell energy density to integrate
					//differentiating with nodal positions leaves terms of force contributions times shape function at the particle locations
					//corresponding to the force contributions at that quadrature point
          /*
          if(update->whichflag==2){
          if (!atomic_flag){
					for (int js = 0; js < nodes_per_element; js++) {
						for (int jj = 0; jj < 3; jj++) {
							if (listindex == iii) {
    					current_nodal_gradients[js][poly_counter][jj] += coefficients*force_contribution[jj] *
    						shape_function(s, t, w, 2, js + 1)/2;
							}
    					//listindex determines if the neighbor virtual atom belongs to the current element or a neighboring element
    					//derivative contributions are zero for jth virtual atoms in other elements that depend on other nodal variables
    					if (listindex <atom->nlocal) {
    						nodal_gradients[listindex][js][poly_grad_scan][jj] -= coefficients*force_contribution[jj] *
    							shape_function(scanning_unit_cell[0], scanning_unit_cell[1], scanning_unit_cell[2], 2, js + 1)/2;
    					}

						}
					}
          }
          else{
            for (int jj = 0; jj < 3; jj++){
                    current_nodal_gradients[0][poly_counter][jj] += coefficients*force_contribution[jj];
                }
          }
        }
        */
				}
				//end of energy portion
			}

		  //portion that computes the energy and gradients of energy
         


    


//end of scanning loop


 //induce segfault to debug
 //segv=force_density[133][209];








}
