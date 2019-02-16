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

#include "npair_CAC.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "atom.h"
#include "atom_vec.h"
#include "molecule.h"
#include "domain.h"
#include "my_page.h"
#include "error.h"
#include "memory.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

NPairCAC::NPairCAC(LAMMPS *lmp) : NPair(lmp) {}

/* ----------------------------------------------------------------------
   binned neighbor list construction for all neighbors
   multi-type stencil is itype dependent and is distance checked
   every neighbor pair appears in list of both atoms i and j
------------------------------------------------------------------------- */

void NPairCAC::build(NeighList *list)
{
	int i, j, k, n, itype, jtype, ibin, which, imol, iatom, moltemplate;
	tagint tagprev;
	double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
	int *neighptr;

	double **x = atom->x;
	int *type = atom->type;
	int *mask = atom->mask;
	tagint *tag = atom->tag;
	tagint *molecule = atom->molecule;
	tagint **special = atom->special;
	int **nspecial = atom->nspecial;
	int nlocal = atom->nlocal;
	if (includegroup) nlocal = atom->nfirst;

	int *molindex = atom->molindex;
	int *molatom = atom->molatom;
	Molecule **onemols = atom->avec->onemols;
	if (molecular == 2) moltemplate = 1;
	else moltemplate = 0;
	CAC_cut = atom->CAC_cut;
	
	
	
	int *ilist = list->ilist;
	int *numneigh = list->numneigh;
	int **firstneigh = list->firstneigh;
	MyPage<int> *ipage = list->ipage;


  int *element_type = atom->element_type;
  element_scale = atom->element_scale;
  int current_element_type;
  int neighbor_element_type;

  quadrature_init(2);
  int inum = 0;
  ipage->reset();

  for (i = 0; i < nlocal; i++) {
    n = 0;
    neighptr = ipage->vget();

    itype = type[i];
	current_element_type = element_type[i];
	current_element_scale[0] = element_scale[i][0];
	current_element_scale[1] = element_scale[i][1];
	current_element_scale[2] = element_scale[i][2];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    if (moltemplate) {
      imol = molindex[i];
      iatom = molatom[i];
      tagprev = tag[i] - iatom - 1;
    }

	// loop over all atoms in surrounding bins in stencil including self
	// skip i = j

	ibin = atom2bin[i];

	for (k = 0; k < nstencil; k++) {
		for (j = binhead[ibin + stencil[k]]; j >= 0; j = bins[j]) {
			if (i == j) continue;

			jtype = type[j];
			
			neighbor_element_type = element_type[j];
		



			if (exclude && exclusion(i, j, itype, jtype, mask, molecule)) continue;

			delx = xtmp - x[j][0];
			dely = ytmp - x[j][1];
			delz = ztmp - x[j][2];
			rsq = delx*delx + dely*dely + delz*delz;
			if (neighbor_element_type != 0&&current_element_type!=0) {
			if(CAC_decide_element2element(i,j)) neighptr[n++] = j;
			}
			else if(neighbor_element_type == 0 && current_element_type != 0){
				if(CAC_decide_element2atom(i,j)) neighptr[n++] = j;
			}
			else if (neighbor_element_type != 0 && current_element_type == 0) {
				if (CAC_decide_atom2element(j, i))  neighptr[n++] = j;
				
			}
			else if(neighbor_element_type == 0 && current_element_type == 0){
				if (rsq <= CAC_cut*CAC_cut) {
					if (molecular) {
						if (!moltemplate)
							which = find_special(special[i], nspecial[i], tag[j]);
						else if (imol >= 0)
							which = find_special(onemols[imol]->special[iatom],
								onemols[imol]->nspecial[iatom],
								tag[j] - tagprev);
						else which = 0;
						if (which == 0) neighptr[n++] = j;
						else if (domain->minimum_image_check(delx, dely, delz))
							neighptr[n++] = j;
						else if (which > 0) neighptr[n++] = j ^ (which << SBBITS);
					}
					else neighptr[n++] = j;
				}

			}
		}
	}

    ilist[inum++] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage->vgot(n);
    if (ipage->status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
  }

  list->inum = inum;
  list->gnum = 0;
  memory->destroy(quadrature_abcissae);
}

///////////////////////////////////////////
double NPairCAC::shape_function(double s, double t, double w, int flag, int index) {
	double shape_function = 0;
	if (flag == 2) {

		if (index == 1) {
			shape_function = (1 - s)*(1 - t)*(1 - w) / 8;
		}
		else if (index == 2) {
			shape_function = (1 + s)*(1 - t)*(1 - w) / 8;
		}
		else if (index == 3) {
			shape_function = (1 + s)*(1 + t)*(1 - w) / 8;
		}
		else if (index == 4) {
			shape_function = (1 - s)*(1 + t)*(1 - w) / 8;
		}
		else if (index == 5) {
			shape_function = (1 - s)*(1 - t)*(1 + w) / 8;
		}
		else if (index == 6) {
			shape_function = (1 + s)*(1 - t)*(1 + w) / 8;
		}
		else if (index == 7) {
			shape_function = (1 + s)*(1 + t)*(1 + w) / 8;
		}
		else if (index == 8) {
			shape_function = (1 - s)*(1 + t)*(1 + w) / 8;
		}


	}
	return shape_function;


}

////////////////////////////////////////////////////////

void NPairCAC::compute_surface_depths(double &scalex, double &scaley, double &scalez,
	int &countx, int &county, int &countz, int flag) {
	int poly = current_poly_counter;
	double unit_cell_mapped[3];
	double rcut;
	rcut = CAC_cut; 
	//flag determines the current element type and corresponding procedure to calculate parameters for 
	//surface penetration depth to be used when computing force density with influences from neighboring
	//elements

	unit_cell_mapped[0] = 2 / double(current_element_scale[0]);
	unit_cell_mapped[1] = 2 / double(current_element_scale[1]);
	unit_cell_mapped[2] = 2 / double(current_element_scale[2]);
	double ds_x = (current_nodal_positions[0][poly][0] - current_nodal_positions[1][poly][0])*
		(current_nodal_positions[0][poly][0] - current_nodal_positions[1][poly][0]);
	double ds_y = (current_nodal_positions[0][poly][1] - current_nodal_positions[1][poly][1])*
		(current_nodal_positions[0][poly][1] - current_nodal_positions[1][poly][1]);
	double ds_z = (current_nodal_positions[0][poly][2] - current_nodal_positions[1][poly][2])*
		(current_nodal_positions[0][poly][2] - current_nodal_positions[1][poly][2]);
	double ds_surf = 2 * rcut / sqrt(ds_x + ds_y + ds_z);
	ds_surf = unit_cell_mapped[0] * (int)(ds_surf / unit_cell_mapped[0]) + unit_cell_mapped[0];

	double dt_x = (current_nodal_positions[0][poly][0] - current_nodal_positions[3][poly][0])*
		(current_nodal_positions[0][poly][0] - current_nodal_positions[3][poly][0]);
	double dt_y = (current_nodal_positions[0][poly][1] - current_nodal_positions[3][poly][1])*
		(current_nodal_positions[0][poly][1] - current_nodal_positions[3][poly][1]);
	double dt_z = (current_nodal_positions[0][poly][2] - current_nodal_positions[3][poly][2])*
		(current_nodal_positions[0][poly][2] - current_nodal_positions[3][poly][2]);

	double dt_surf = 2 * rcut / sqrt(dt_x + dt_y + dt_z);
	dt_surf = unit_cell_mapped[1] * (int)(dt_surf / unit_cell_mapped[1]) + unit_cell_mapped[1];

	double dw_x = (current_nodal_positions[0][poly][0] - current_nodal_positions[4][poly][0])*
		(current_nodal_positions[0][poly][0] - current_nodal_positions[4][poly][0]);
	double dw_y = (current_nodal_positions[0][poly][1] - current_nodal_positions[4][poly][1])*
		(current_nodal_positions[0][poly][1] - current_nodal_positions[3][poly][1]);
	double dw_z = (current_nodal_positions[0][poly][2] - current_nodal_positions[4][poly][2])*
		(current_nodal_positions[0][poly][2] - current_nodal_positions[4][poly][2]);

	double dw_surf = 2 * rcut / sqrt(dw_x + dw_y + dw_z);
	dw_surf = unit_cell_mapped[2] * (int)(dw_surf / unit_cell_mapped[2]) + unit_cell_mapped[2];
	if (ds_surf > 1) {
		ds_surf = 1;


	}
	if (dt_surf > 1) {

		dt_surf = 1;


	}
	if (dw_surf > 1) {

		dw_surf = 1;

	}
	scalex = 1 - ds_surf;
	scaley = 1 - dt_surf;
	scalez = 1 - dw_surf;

	countx = (int)(ds_surf / unit_cell_mapped[0]);
	county = (int)(dt_surf / unit_cell_mapped[1]);
	countz = (int)(dw_surf / unit_cell_mapped[2]);




}


/////////////////////////////////


void NPairCAC::quadrature_init(int quadrature_rank) {

	if (quadrature_rank == 1) {
		quadrature_node_count = 1;
		
		memory->create(quadrature_abcissae, quadrature_node_count, "pairCAC:quadrature_abcissae");

		quadrature_abcissae[0] = 0;
	}
	if (quadrature_rank == 2) {



		quadrature_node_count = 2;

		memory->create(quadrature_abcissae, quadrature_node_count, "pairCAC:quadrature_abcissae");

		quadrature_abcissae[0] = -0.5773502691896258;
		quadrature_abcissae[1] = 0.5773502691896258;

	}

	if (quadrature_rank == 3)
	{


	}
	if (quadrature_rank == 4)
	{


	}
	if (quadrature_rank == 5)
	{


	}


}

//decide if an element is close enough to an atom to consider for nonlocal quadrature calculation

int NPairCAC::CAC_decide_element2atom(int element_index, int atom_index){

	double unit_cell_mapped[3];
	double interior_scale[3];
	int surface_counts[3];
	int nodes_per_element;
	int found_flag=0;
	double s, t, w;
	double **x = atom->x;
	s = t = w = 0;
	double sq, tq, wq;
	double quad_position[3];
	double ****nodal_positions = atom->nodal_positions;
	double shape_func;
	int *element_type = atom->element_type;
	int *poly_count = atom->poly_count;
	double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
	int *nodes_per_element_list = atom->nodes_per_element_list;
	 nodes_per_element = nodes_per_element_list[element_type[element_index]];
	unit_cell_mapped[0] = 2 / double(element_scale[element_index][0]);
	unit_cell_mapped[1] = 2 / double(element_scale[element_index][1]);
	unit_cell_mapped[2] = 2 / double(element_scale[element_index][2]);
	current_nodal_positions = nodal_positions[element_index];
	int current_poly_count = poly_count[element_index];



	//compute quadrature point positions to test for neighboring







	int sign[2];
	sign[0] = -1;
	sign[1] = 1;
	

	
	for (int poly_counter = 0; poly_counter < current_poly_count; poly_counter++) {
		current_poly_counter = poly_counter;
		compute_surface_depths(interior_scale[0], interior_scale[1], interior_scale[2],
			surface_counts[0], surface_counts[1], surface_counts[2], 1);

		//interior contributions


		for (int i = 0; i < quadrature_node_count; i++) {
			for (int j = 0; j < quadrature_node_count; j++) {
				for (int k = 0; k < quadrature_node_count; k++) {

					sq = s = interior_scale[0] * quadrature_abcissae[i];
					tq = t = interior_scale[1] * quadrature_abcissae[j];
					wq = w = interior_scale[2] * quadrature_abcissae[k];
					s = unit_cell_mapped[0] * (int(s / unit_cell_mapped[0]));
					t = unit_cell_mapped[1] * (int(t / unit_cell_mapped[1]));
					w = unit_cell_mapped[2] * (int(w / unit_cell_mapped[2]));


					quad_position[0] = 0;
					quad_position[1] = 0;
					quad_position[2] = 0;
					for (int kkk = 0; kkk < nodes_per_element; kkk++) {
						shape_func = shape_function(s, t, w, 2, kkk + 1);
						quad_position[0] += current_nodal_positions[kkk][poly_counter][0] * shape_func;
						quad_position[1] += current_nodal_positions[kkk][poly_counter][1] * shape_func;
						quad_position[2] += current_nodal_positions[kkk][poly_counter][2] * shape_func;
					}

					delx = quad_position[0] - x[atom_index][0];
					dely = quad_position[1] - x[atom_index][1];
					delz = quad_position[2] - x[atom_index][2];
					rsq = delx*delx + dely*dely + delz*delz;
					if (rsq < CAC_cut*CAC_cut) {
						found_flag = 1;
						return found_flag;
					}



				}
			}
		}

		// s axis surface contributions
	for (int sc = 0; sc < 2; sc++) {
		for (int i = 0; i < surface_counts[0]; i++) {
			for (int j = 0; j < quadrature_node_count; j++) {
				for (int k = 0; k < quadrature_node_count; k++) {
					
						s = sign[sc] - i*unit_cell_mapped[0] * sign[sc];

						s = s - 0.5*unit_cell_mapped[0] * sign[sc];
						tq = t = interior_scale[1] * quadrature_abcissae[j];
						wq = w = interior_scale[2] * quadrature_abcissae[k];
						t = unit_cell_mapped[1] * (int(t / unit_cell_mapped[1]));
						w = unit_cell_mapped[2] * (int(w / unit_cell_mapped[2]));

						if (quadrature_abcissae[k] < 0)
							w = w - 0.5*unit_cell_mapped[2];
						else
							w = w + 0.5*unit_cell_mapped[2];

						if (quadrature_abcissae[j] < 0)
							t = t - 0.5*unit_cell_mapped[1];
						else
							t = t + 0.5*unit_cell_mapped[1];

						quad_position[0] = 0;
						quad_position[1] = 0;
						quad_position[2] = 0;
						for (int kkk = 0; kkk < nodes_per_element; kkk++) {
							shape_func = shape_function(s, t, w, 2, kkk + 1);
							quad_position[0] += current_nodal_positions[kkk][poly_counter][0] * shape_func;
							quad_position[1] += current_nodal_positions[kkk][poly_counter][1] * shape_func;
							quad_position[2] += current_nodal_positions[kkk][poly_counter][2] * shape_func;
						}

						delx = quad_position[0] - x[atom_index][0];
						dely = quad_position[1] - x[atom_index][1];
						delz = quad_position[2] - x[atom_index][2];
						rsq = delx*delx + dely*dely + delz*delz;
						if (rsq < CAC_cut*CAC_cut) {
							found_flag = 1;
							return found_flag;
						}
					}
				}
			}
		}
	

	// t axis contributions
	
	for (int sc = 0; sc < 2; sc++) {
		for (int i = 0; i < surface_counts[1]; i++) {
			for (int j = 0; j < quadrature_node_count; j++) {
				for (int k = 0; k < quadrature_node_count; k++) {
					

						sq = s = interior_scale[0] * quadrature_abcissae[j];
						s = unit_cell_mapped[0] * (int(s / unit_cell_mapped[0]));
						t = sign[sc] - i*unit_cell_mapped[1] * sign[sc];

						t = t - 0.5*unit_cell_mapped[1] * sign[sc];
						wq = w = interior_scale[2] * quadrature_abcissae[k];
						w = unit_cell_mapped[2] * (int(w / unit_cell_mapped[2]));

						if (quadrature_abcissae[j] < 0)
							s = s - 0.5*unit_cell_mapped[0];
						else
							s = s + 0.5*unit_cell_mapped[0];

						if (quadrature_abcissae[k] < 0)
							w = w - 0.5*unit_cell_mapped[2];
						else
							w = w + 0.5*unit_cell_mapped[2];

						quad_position[0] = 0;
						quad_position[1] = 0;
						quad_position[2] = 0;
						for (int kkk = 0; kkk < nodes_per_element; kkk++) {
							shape_func = shape_function(s, t, w, 2, kkk + 1);
							quad_position[0] += current_nodal_positions[kkk][poly_counter][0] * shape_func;
							quad_position[1] += current_nodal_positions[kkk][poly_counter][1] * shape_func;
							quad_position[2] += current_nodal_positions[kkk][poly_counter][2] * shape_func;
						}

						delx = quad_position[0] - x[atom_index][0];
						dely = quad_position[1] - x[atom_index][1];
						delz = quad_position[2] - x[atom_index][2];
						rsq = delx*delx + dely*dely + delz*delz;
						if (rsq < CAC_cut*CAC_cut) {
							found_flag = 1;
							return found_flag;
						}

					}
				}
			}
		}
	

	//w axis surface contributions
	
	for (int sc = 0; sc < 2; sc++) {
		for (int i = 0; i < surface_counts[2]; i++) {
			for (int j = 0; j < quadrature_node_count; j++) {
				for (int k = 0; k < quadrature_node_count; k++) {
					

						sq = s = interior_scale[0] * quadrature_abcissae[j];
						s = unit_cell_mapped[0] * (int(s / unit_cell_mapped[0]));
						tq = t = interior_scale[1] * quadrature_abcissae[k];
						t = unit_cell_mapped[1] * (int(t / unit_cell_mapped[1]));
						w = sign[sc] - i*unit_cell_mapped[2] * sign[sc];

						w = w - 0.5*unit_cell_mapped[2] * sign[sc];

						if (quadrature_abcissae[j] < 0)
							s = s - 0.5*unit_cell_mapped[0];
						else
							s = s + 0.5*unit_cell_mapped[0];

						if (quadrature_abcissae[k] < 0)
							t = t - 0.5*unit_cell_mapped[1];
						else
							t = t + 0.5*unit_cell_mapped[1];


						quad_position[0] = 0;
						quad_position[1] = 0;
						quad_position[2] = 0;
						for (int kkk = 0; kkk < nodes_per_element; kkk++) {
							shape_func = shape_function(s, t, w, 2, kkk + 1);
							quad_position[0] += current_nodal_positions[kkk][poly_counter][0] * shape_func;
							quad_position[1] += current_nodal_positions[kkk][poly_counter][1] * shape_func;
							quad_position[2] += current_nodal_positions[kkk][poly_counter][2] * shape_func;
						}

						delx = quad_position[0] - x[atom_index][0];
						dely = quad_position[1] - x[atom_index][1];
						delz = quad_position[2] - x[atom_index][2];
						rsq = delx*delx + dely*dely + delz*delz;
						if (rsq < CAC_cut*CAC_cut) {
							found_flag = 1;
							return found_flag;
						}

					}
				}
			}
		}
	


	int surface_countsx;
	int surface_countsy;

	//compute edge contributions

	for (int sc = 0; sc < 12; sc++) {
		if (sc == 0 || sc == 1 || sc == 2 || sc == 3) {

			surface_countsx = surface_counts[0];
			surface_countsy = surface_counts[1];
		}
		else if (sc == 4 || sc == 5 || sc == 6 || sc == 7) {

			surface_countsx = surface_counts[1];
			surface_countsy = surface_counts[2];
		}
		else if (sc == 8 || sc == 9 || sc == 10 || sc == 11) {

			surface_countsx = surface_counts[0];
			surface_countsy = surface_counts[2];
		}


		
		for (int i = 0; i < surface_countsx; i++) {//alter surface counts for specific corner
			for (int j = 0; j < surface_countsy; j++) {
				
				for (int k = 0; k < quadrature_node_count; k++) {
					
						if (sc == 0) {

							sq = s = -1 + (i + 0.5)*unit_cell_mapped[0];
							tq = t = -1 + (j + 0.5)*unit_cell_mapped[1];
							wq = w = interior_scale[2] * quadrature_abcissae[k];
							w = unit_cell_mapped[2] * (int(w / unit_cell_mapped[2]));
							if (quadrature_abcissae[k] < 0)
								w = w - 0.5*unit_cell_mapped[2];
							else
								w = w + 0.5*unit_cell_mapped[2];
						}
						else if (sc == 1) {
							sq = s = 1 - (i + 0.5)*unit_cell_mapped[0];
							tq = t = -1 + (j + 0.5)*unit_cell_mapped[1];
							wq = w = interior_scale[2] * quadrature_abcissae[k];
							w = unit_cell_mapped[2] * (int(w / unit_cell_mapped[2]));
							if (quadrature_abcissae[k] < 0)
								w = w - 0.5*unit_cell_mapped[2];
							else
								w = w + 0.5*unit_cell_mapped[2];
						}
						else if (sc == 2) {
							sq = s = -1 + (i + 0.5)*unit_cell_mapped[0];
							tq = t = 1 - (j + 0.5)*unit_cell_mapped[1];
							wq = w = interior_scale[2] * quadrature_abcissae[k];
							w = unit_cell_mapped[2] * (int(w / unit_cell_mapped[2]));
							if (quadrature_abcissae[k] < 0)
								w = w - 0.5*unit_cell_mapped[2];
							else
								w = w + 0.5*unit_cell_mapped[2];
						}
						else if (sc == 3) {
							sq = s = 1 - (i + 0.5)*unit_cell_mapped[0];
							tq = t = 1 - (j + 0.5)*unit_cell_mapped[1];
							wq = w = interior_scale[2] * quadrature_abcissae[k];
							w = unit_cell_mapped[2] * (int(w / unit_cell_mapped[2]));
							if (quadrature_abcissae[k] < 0)
								w = w - 0.5*unit_cell_mapped[2];
							else
								w = w + 0.5*unit_cell_mapped[2];
						}
						else if (sc == 4) {
							sq = s = interior_scale[0] * quadrature_abcissae[k];
							s = unit_cell_mapped[0] * (int(s / unit_cell_mapped[0]));
							tq = t = -1 + (i + 0.5)*unit_cell_mapped[1];
							wq = w = -1 + (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								s = s - 0.5*unit_cell_mapped[0];
							else
								s = s + 0.5*unit_cell_mapped[0];

						}
						else if (sc == 5) {
							sq = s = interior_scale[0] * quadrature_abcissae[k];
							s = unit_cell_mapped[0] * (int(s / unit_cell_mapped[0]));
							tq = t = 1 - (i + 0.5)*unit_cell_mapped[1];
							wq = w = -1 + (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								s = s - 0.5*unit_cell_mapped[0];
							else
								s = s + 0.5*unit_cell_mapped[0];
						}
						else if (sc == 6) {
							sq = s = interior_scale[0] * quadrature_abcissae[k];
							s = unit_cell_mapped[0] * (int(s / unit_cell_mapped[0]));
							tq = t = -1 + (i + 0.5)*unit_cell_mapped[1];
							wq = w = 1 - (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								s = s - 0.5*unit_cell_mapped[0];
							else
								s = s + 0.5*unit_cell_mapped[0];
						}
						else if (sc == 7) {
							sq = s = interior_scale[0] * quadrature_abcissae[k];
							s = unit_cell_mapped[0] * (int(s / unit_cell_mapped[0]));
							tq = t = 1 - (i + 0.5)*unit_cell_mapped[1];
							wq = w = 1 - (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								s = s - 0.5*unit_cell_mapped[0];
							else
								s = s + 0.5*unit_cell_mapped[0];
						}
						else if (sc == 8) {
							sq = s = -1 + (i + 0.5)*unit_cell_mapped[0];
							tq = t = interior_scale[1] * quadrature_abcissae[k];
							t = unit_cell_mapped[1] * (int(t / unit_cell_mapped[1]));
							wq = w = -1 + (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								t = t - 0.5*unit_cell_mapped[1];
							else
								t = t + 0.5*unit_cell_mapped[1];

						}
						else if (sc == 9) {
							sq = s = 1 - (i + 0.5)*unit_cell_mapped[0];
							tq = t = interior_scale[1] * quadrature_abcissae[k];
							t = unit_cell_mapped[1] * (int(t / unit_cell_mapped[1]));
							wq = w = -1 + (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								t = t - 0.5*unit_cell_mapped[1];
							else
								t = t + 0.5*unit_cell_mapped[1];
						}
						else if (sc == 10) {
							sq = s = -1 + (i + 0.5)*unit_cell_mapped[0];
							tq = t = interior_scale[1] * quadrature_abcissae[k];
							t = unit_cell_mapped[1] * (int(t / unit_cell_mapped[1]));
							wq = w = 1 - (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								t = t - 0.5*unit_cell_mapped[1];
							else
								t = t + 0.5*unit_cell_mapped[1];
						}
						else if (sc == 11) {
							sq = s = 1 - (i + 0.5)*unit_cell_mapped[0];
							tq = t = interior_scale[1] * quadrature_abcissae[k];
							t = unit_cell_mapped[1] * (int(t / unit_cell_mapped[1]));
							wq = w = 1 - (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								t = t - 0.5*unit_cell_mapped[1];
							else
								t = t + 0.5*unit_cell_mapped[1];
						}



						quad_position[0] = 0;
						quad_position[1] = 0;
						quad_position[2] = 0;
						for (int kkk = 0; kkk < nodes_per_element; kkk++) {
							shape_func = shape_function(s, t, w, 2, kkk + 1);
							quad_position[0] += current_nodal_positions[kkk][poly_counter][0] * shape_func;
							quad_position[1] += current_nodal_positions[kkk][poly_counter][1] * shape_func;
							quad_position[2] += current_nodal_positions[kkk][poly_counter][2] * shape_func;
						}

						delx = quad_position[0] - x[atom_index][0];
						dely = quad_position[1] - x[atom_index][1];
						delz = quad_position[2] - x[atom_index][2];
						rsq = delx*delx + dely*dely + delz*delz;
						if (rsq < CAC_cut*CAC_cut) {
							found_flag = 1;
							return found_flag;
						}
					}

				}

			}
		}
	


	//compute corner contributions

	for (int sc = 0; sc < 8; sc++) {
		for (int i = 0; i < surface_counts[0]; i++) {//alter surface counts for specific corner
			for (int j = 0; j < surface_counts[1]; j++) {
				for (int k = 0; k < surface_counts[2]; k++) {
					
					
						if (sc == 0) {

							s = -1 + (i + 0.5)*unit_cell_mapped[0];
							t = -1 + (j + 0.5)*unit_cell_mapped[1];
							w = -1 + (k + 0.5)*unit_cell_mapped[2];

						}
						else if (sc == 1) {
							s = 1 - (i + 0.5)*unit_cell_mapped[0];
							t = -1 + (j + 0.5)*unit_cell_mapped[1];
							w = -1 + (k + 0.5)*unit_cell_mapped[2];
						}
						else if (sc == 2) {
							s = 1 - (i + 0.5)*unit_cell_mapped[0];
							t = 1 - (j + 0.5)*unit_cell_mapped[1];
							w = -1 + (k + 0.5)*unit_cell_mapped[2];
						}
						else if (sc == 3) {
							s = -1 + (i + 0.5)*unit_cell_mapped[0];
							t = 1 - (j + 0.5)*unit_cell_mapped[1];
							w = -1 + (k + 0.5)*unit_cell_mapped[2];
						}
						else if (sc == 4) {
							s = -1 + (i + 0.5)*unit_cell_mapped[0];
							t = -1 + (j + 0.5)*unit_cell_mapped[1];
							w = 1 - (k + 0.5)*unit_cell_mapped[2];

						}
						else if (sc == 5) {
							s = 1 - (i + 0.5)*unit_cell_mapped[0];
							t = -1 + (j + 0.5)*unit_cell_mapped[1];
							w = 1 - (k + 0.5)*unit_cell_mapped[2];
						}
						else if (sc == 6) {
							s = 1 - (i + 0.5)*unit_cell_mapped[0];
							t = 1 - (j + 0.5)*unit_cell_mapped[1];
							w = 1 - (k + 0.5)*unit_cell_mapped[2];
						}
						else if (sc == 7) {
							s = -1 + (i + 0.5)*unit_cell_mapped[0];
							t = 1 - (j + 0.5)*unit_cell_mapped[1];
							w = 1 - (k + 0.5)*unit_cell_mapped[2];
						}

						quad_position[0] = 0;
						quad_position[1] = 0;
						quad_position[2] = 0;
						for (int kkk = 0; kkk < nodes_per_element; kkk++) {
							shape_func = shape_function(s, t, w, 2, kkk + 1);
							quad_position[0] += current_nodal_positions[kkk][poly_counter][0] * shape_func;
							quad_position[1] += current_nodal_positions[kkk][poly_counter][1] * shape_func;
							quad_position[2] += current_nodal_positions[kkk][poly_counter][2] * shape_func;
						}

						delx = quad_position[0] - x[atom_index][0];
						dely = quad_position[1] - x[atom_index][1];
						delz = quad_position[2] - x[atom_index][2];
						rsq = delx*delx + dely*dely + delz*delz;
						if (rsq < CAC_cut*CAC_cut) {
							found_flag = 1;
							return found_flag;
						}

					}
				}

			}
		}
	}


	return found_flag;

}

//decide if an atom is close enough to an element to consider for nonlocal quadrature calculation

int NPairCAC::CAC_decide_atom2element(int element_index, int atom_index) {
	double unit_cell_mapped[3];
	double interior_scale[3];
	int surface_counts[3];
	int nodes_per_element;
	int found_flag = 0;
	double s, t, w;
	double **x = atom->x;
	s = t = w = 0;
	double sq, tq, wq;
	double quad_position[3];
	double ****nodal_positions = atom->nodal_positions;
	double shape_func;
	int *element_type = atom->element_type;
	int *poly_count = atom->poly_count;
	double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
	double bounding_boxlo[3];
	double bounding_boxhi[3];
	int *nodes_per_element_list = atom->nodes_per_element_list;
	nodes_per_element = nodes_per_element_list[element_type[element_index]];
	unit_cell_mapped[0] = 2 / double(element_scale[element_index][0]);
	unit_cell_mapped[1] = 2 / double(element_scale[element_index][1]);
	unit_cell_mapped[2] = 2 / double(element_scale[element_index][2]);
	current_nodal_positions = nodal_positions[element_index];
	//initialize bounding box values
	bounding_boxlo[0] = current_nodal_positions[0][0][0];
	bounding_boxlo[1] = current_nodal_positions[0][0][1];
	bounding_boxlo[2] = current_nodal_positions[0][0][2];
	bounding_boxhi[0] = current_nodal_positions[0][0][0];
	bounding_boxhi[1] = current_nodal_positions[0][0][1];
	bounding_boxhi[2] = current_nodal_positions[0][0][2];

	int current_poly_count = poly_count[element_index];
	for (int poly_counter = 0; poly_counter < current_poly_count; poly_counter++) {
		for (int kkk = 0; kkk < nodes_per_element; kkk++) {
			for (int dim = 0; dim < 3; dim++) {
				if (current_nodal_positions[kkk][poly_counter][dim] < bounding_boxlo[dim]) {
					bounding_boxlo[dim] = current_nodal_positions[kkk][poly_counter][dim];
				}
				if (current_nodal_positions[kkk][poly_counter][dim] > bounding_boxhi[dim]) {
					bounding_boxhi[dim] = current_nodal_positions[kkk][poly_counter][dim];
				}
			}
		}
	}

	bounding_boxlo[0] -= CAC_cut;
	bounding_boxlo[1] -= CAC_cut;
	bounding_boxlo[2] -= CAC_cut;
	bounding_boxhi[0] += CAC_cut;
	bounding_boxhi[1] += CAC_cut;
	bounding_boxhi[2] += CAC_cut;
	if(x[atom_index][0]>bounding_boxlo[0]&&x[atom_index][0]<bounding_boxhi[0]&&
		x[atom_index][1]>bounding_boxlo[1]&&x[atom_index][1]<bounding_boxhi[1]&&
		x[atom_index][2]>bounding_boxlo[2]&&x[atom_index][2]<bounding_boxhi[2])
	{
		found_flag = 1;
		return found_flag;
	}
	else {
		return found_flag;
	}
	

}

//decide if an element is close enough to another element to consider for nonlocal quadrature calculation

int NPairCAC::CAC_decide_element2element(int element_index, int neighbor_element_index) {


    double unit_cell_mapped[3];
	double interior_scale[3];
	int surface_counts[3];
	
	int found_flag=0;
	double s, t, w;
	double **x = atom->x;
	s = t = w = 0;
	double sq, tq, wq;
	double quad_position[3];
	double ****nodal_positions = atom->nodal_positions;
	double shape_func;
	int *element_type = atom->element_type;
	int *poly_count = atom->poly_count;
	double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
	int *nodes_per_element_list = atom->nodes_per_element_list;
	int nodes_per_element = nodes_per_element_list[element_type[element_index]];
	unit_cell_mapped[0] = 2 / double(element_scale[element_index][0]);
	unit_cell_mapped[1] = 2 / double(element_scale[element_index][1]);
	unit_cell_mapped[2] = 2 / double(element_scale[element_index][2]);
	current_nodal_positions = nodal_positions[element_index];
	double ***neighbor_nodal_positions=nodal_positions[neighbor_element_index];
	int current_poly_count = poly_count[element_index];
    int neighbor_poly_count = poly_count[neighbor_element_index];
	int neighbor_nodes_per_element = nodes_per_element_list[element_type[neighbor_element_index]];
 


	double bounding_boxlo[3];
	double bounding_boxhi[3];

	//initialize bounding box values
	bounding_boxlo[0] = neighbor_nodal_positions[0][0][0];
	bounding_boxlo[1] = neighbor_nodal_positions[0][0][1];
	bounding_boxlo[2] = neighbor_nodal_positions[0][0][2];
	bounding_boxhi[0] = neighbor_nodal_positions[0][0][0];
	bounding_boxhi[1] = neighbor_nodal_positions[0][0][1];
	bounding_boxhi[2] = neighbor_nodal_positions[0][0][2];
     //define the bounding box for the element being considered as a neighbor
	
	for (int poly_counter = 0; poly_counter < neighbor_poly_count; poly_counter++) {
		for (int kkk = 0; kkk < neighbor_nodes_per_element; kkk++) {
			for (int dim = 0; dim < 3; dim++) {
				if (neighbor_nodal_positions[kkk][poly_counter][dim] < bounding_boxlo[dim]) {
					bounding_boxlo[dim] = neighbor_nodal_positions[kkk][poly_counter][dim];
				}
				if (neighbor_nodal_positions[kkk][poly_counter][dim] > bounding_boxhi[dim]) {
					bounding_boxhi[dim] = neighbor_nodal_positions[kkk][poly_counter][dim];
				}
			}
		}
	}

	bounding_boxlo[0] -= CAC_cut;
	bounding_boxlo[1] -= CAC_cut;
	bounding_boxlo[2] -= CAC_cut;
	bounding_boxhi[0] += CAC_cut;
	bounding_boxhi[1] += CAC_cut;
	bounding_boxhi[2] += CAC_cut;

	//compute quadrature point positions to test for neighboring
	//if any quadrature point is inside the box 

	int sign[2];
	sign[0] = -1;
	sign[1] = 1;
	

	
	for (int poly_counter = 0; poly_counter < current_poly_count; poly_counter++) {
		current_poly_counter = poly_counter;
		compute_surface_depths(interior_scale[0], interior_scale[1], interior_scale[2],
			surface_counts[0], surface_counts[1], surface_counts[2], 1);

		//interior contributions


		for (int i = 0; i < quadrature_node_count; i++) {
			for (int j = 0; j < quadrature_node_count; j++) {
				for (int k = 0; k < quadrature_node_count; k++) {

					sq = s = interior_scale[0] * quadrature_abcissae[i];
					tq = t = interior_scale[1] * quadrature_abcissae[j];
					wq = w = interior_scale[2] * quadrature_abcissae[k];
					s = unit_cell_mapped[0] * (int(s / unit_cell_mapped[0]));
					t = unit_cell_mapped[1] * (int(t / unit_cell_mapped[1]));
					w = unit_cell_mapped[2] * (int(w / unit_cell_mapped[2]));


					quad_position[0] = 0;
					quad_position[1] = 0;
					quad_position[2] = 0;
					for (int kkk = 0; kkk < nodes_per_element; kkk++) {
						shape_func = shape_function(s, t, w, 2, kkk + 1);
						quad_position[0] += current_nodal_positions[kkk][poly_counter][0] * shape_func;
						quad_position[1] += current_nodal_positions[kkk][poly_counter][1] * shape_func;
						quad_position[2] += current_nodal_positions[kkk][poly_counter][2] * shape_func;
					}

					if(quad_position[0]>bounding_boxlo[0]&&quad_position[0]<bounding_boxhi[0]&&
					quad_position[1]>bounding_boxlo[1]&&quad_position[1]<bounding_boxhi[1]&&
					quad_position[2]>bounding_boxlo[2]&&quad_position[2]<bounding_boxhi[2]) {
						found_flag = 1;
						return found_flag;
					}



				}
			}
		}

		// s axis surface contributions
	for (int sc = 0; sc < 2; sc++) {
		for (int i = 0; i < surface_counts[0]; i++) {
			for (int j = 0; j < quadrature_node_count; j++) {
				for (int k = 0; k < quadrature_node_count; k++) {
					
						s = sign[sc] - i*unit_cell_mapped[0] * sign[sc];

						s = s - 0.5*unit_cell_mapped[0] * sign[sc];
						tq = t = interior_scale[1] * quadrature_abcissae[j];
						wq = w = interior_scale[2] * quadrature_abcissae[k];
						t = unit_cell_mapped[1] * (int(t / unit_cell_mapped[1]));
						w = unit_cell_mapped[2] * (int(w / unit_cell_mapped[2]));

						if (quadrature_abcissae[k] < 0)
							w = w - 0.5*unit_cell_mapped[2];
						else
							w = w + 0.5*unit_cell_mapped[2];

						if (quadrature_abcissae[j] < 0)
							t = t - 0.5*unit_cell_mapped[1];
						else
							t = t + 0.5*unit_cell_mapped[1];

						quad_position[0] = 0;
						quad_position[1] = 0;
						quad_position[2] = 0;
						for (int kkk = 0; kkk < nodes_per_element; kkk++) {
							shape_func = shape_function(s, t, w, 2, kkk + 1);
							quad_position[0] += current_nodal_positions[kkk][poly_counter][0] * shape_func;
							quad_position[1] += current_nodal_positions[kkk][poly_counter][1] * shape_func;
							quad_position[2] += current_nodal_positions[kkk][poly_counter][2] * shape_func;
						}

				if(quad_position[0]>bounding_boxlo[0]&&quad_position[0]<bounding_boxhi[0]&&
					quad_position[1]>bounding_boxlo[1]&&quad_position[1]<bounding_boxhi[1]&&
					quad_position[2]>bounding_boxlo[2]&&quad_position[2]<bounding_boxhi[2]) {
						found_flag = 1;
						return found_flag;
					}
					}
				}
			}
		}
	

	// t axis contributions
	
	for (int sc = 0; sc < 2; sc++) {
		for (int i = 0; i < surface_counts[1]; i++) {
			for (int j = 0; j < quadrature_node_count; j++) {
				for (int k = 0; k < quadrature_node_count; k++) {
					

						sq = s = interior_scale[0] * quadrature_abcissae[j];
						s = unit_cell_mapped[0] * (int(s / unit_cell_mapped[0]));
						t = sign[sc] - i*unit_cell_mapped[1] * sign[sc];

						t = t - 0.5*unit_cell_mapped[1] * sign[sc];
						wq = w = interior_scale[2] * quadrature_abcissae[k];
						w = unit_cell_mapped[2] * (int(w / unit_cell_mapped[2]));

						if (quadrature_abcissae[j] < 0)
							s = s - 0.5*unit_cell_mapped[0];
						else
							s = s + 0.5*unit_cell_mapped[0];

						if (quadrature_abcissae[k] < 0)
							w = w - 0.5*unit_cell_mapped[2];
						else
							w = w + 0.5*unit_cell_mapped[2];

						quad_position[0] = 0;
						quad_position[1] = 0;
						quad_position[2] = 0;
						for (int kkk = 0; kkk < nodes_per_element; kkk++) {
							shape_func = shape_function(s, t, w, 2, kkk + 1);
							quad_position[0] += current_nodal_positions[kkk][poly_counter][0] * shape_func;
							quad_position[1] += current_nodal_positions[kkk][poly_counter][1] * shape_func;
							quad_position[2] += current_nodal_positions[kkk][poly_counter][2] * shape_func;
						}

					if(quad_position[0]>bounding_boxlo[0]&&quad_position[0]<bounding_boxhi[0]&&
					quad_position[1]>bounding_boxlo[1]&&quad_position[1]<bounding_boxhi[1]&&
					quad_position[2]>bounding_boxlo[2]&&quad_position[2]<bounding_boxhi[2]) {
						found_flag = 1;
						return found_flag;
					}

					}
				}
			}
		}
	

	//w axis surface contributions
	
	for (int sc = 0; sc < 2; sc++) {
		for (int i = 0; i < surface_counts[2]; i++) {
			for (int j = 0; j < quadrature_node_count; j++) {
				for (int k = 0; k < quadrature_node_count; k++) {
					

						sq = s = interior_scale[0] * quadrature_abcissae[j];
						s = unit_cell_mapped[0] * (int(s / unit_cell_mapped[0]));
						tq = t = interior_scale[1] * quadrature_abcissae[k];
						t = unit_cell_mapped[1] * (int(t / unit_cell_mapped[1]));
						w = sign[sc] - i*unit_cell_mapped[2] * sign[sc];

						w = w - 0.5*unit_cell_mapped[2] * sign[sc];

						if (quadrature_abcissae[j] < 0)
							s = s - 0.5*unit_cell_mapped[0];
						else
							s = s + 0.5*unit_cell_mapped[0];

						if (quadrature_abcissae[k] < 0)
							t = t - 0.5*unit_cell_mapped[1];
						else
							t = t + 0.5*unit_cell_mapped[1];


						quad_position[0] = 0;
						quad_position[1] = 0;
						quad_position[2] = 0;
						for (int kkk = 0; kkk < nodes_per_element; kkk++) {
							shape_func = shape_function(s, t, w, 2, kkk + 1);
							quad_position[0] += current_nodal_positions[kkk][poly_counter][0] * shape_func;
							quad_position[1] += current_nodal_positions[kkk][poly_counter][1] * shape_func;
							quad_position[2] += current_nodal_positions[kkk][poly_counter][2] * shape_func;
						}

						if(quad_position[0]>bounding_boxlo[0]&&quad_position[0]<bounding_boxhi[0]&&
					quad_position[1]>bounding_boxlo[1]&&quad_position[1]<bounding_boxhi[1]&&
					quad_position[2]>bounding_boxlo[2]&&quad_position[2]<bounding_boxhi[2]) {
						found_flag = 1;
						return found_flag;
					}

					}
				}
			}
		}
	


	int surface_countsx;
	int surface_countsy;

	//compute edge contributions

	for (int sc = 0; sc < 12; sc++) {
		if (sc == 0 || sc == 1 || sc == 2 || sc == 3) {

			surface_countsx = surface_counts[0];
			surface_countsy = surface_counts[1];
		}
		else if (sc == 4 || sc == 5 || sc == 6 || sc == 7) {

			surface_countsx = surface_counts[1];
			surface_countsy = surface_counts[2];
		}
		else if (sc == 8 || sc == 9 || sc == 10 || sc == 11) {

			surface_countsx = surface_counts[0];
			surface_countsy = surface_counts[2];
		}


		
		for (int i = 0; i < surface_countsx; i++) {//alter surface counts for specific corner
			for (int j = 0; j < surface_countsy; j++) {
				
				for (int k = 0; k < quadrature_node_count; k++) {
					
						if (sc == 0) {

							sq = s = -1 + (i + 0.5)*unit_cell_mapped[0];
							tq = t = -1 + (j + 0.5)*unit_cell_mapped[1];
							wq = w = interior_scale[2] * quadrature_abcissae[k];
							w = unit_cell_mapped[2] * (int(w / unit_cell_mapped[2]));
							if (quadrature_abcissae[k] < 0)
								w = w - 0.5*unit_cell_mapped[2];
							else
								w = w + 0.5*unit_cell_mapped[2];
						}
						else if (sc == 1) {
							sq = s = 1 - (i + 0.5)*unit_cell_mapped[0];
							tq = t = -1 + (j + 0.5)*unit_cell_mapped[1];
							wq = w = interior_scale[2] * quadrature_abcissae[k];
							w = unit_cell_mapped[2] * (int(w / unit_cell_mapped[2]));
							if (quadrature_abcissae[k] < 0)
								w = w - 0.5*unit_cell_mapped[2];
							else
								w = w + 0.5*unit_cell_mapped[2];
						}
						else if (sc == 2) {
							sq = s = -1 + (i + 0.5)*unit_cell_mapped[0];
							tq = t = 1 - (j + 0.5)*unit_cell_mapped[1];
							wq = w = interior_scale[2] * quadrature_abcissae[k];
							w = unit_cell_mapped[2] * (int(w / unit_cell_mapped[2]));
							if (quadrature_abcissae[k] < 0)
								w = w - 0.5*unit_cell_mapped[2];
							else
								w = w + 0.5*unit_cell_mapped[2];
						}
						else if (sc == 3) {
							sq = s = 1 - (i + 0.5)*unit_cell_mapped[0];
							tq = t = 1 - (j + 0.5)*unit_cell_mapped[1];
							wq = w = interior_scale[2] * quadrature_abcissae[k];
							w = unit_cell_mapped[2] * (int(w / unit_cell_mapped[2]));
							if (quadrature_abcissae[k] < 0)
								w = w - 0.5*unit_cell_mapped[2];
							else
								w = w + 0.5*unit_cell_mapped[2];
						}
						else if (sc == 4) {
							sq = s = interior_scale[0] * quadrature_abcissae[k];
							s = unit_cell_mapped[0] * (int(s / unit_cell_mapped[0]));
							tq = t = -1 + (i + 0.5)*unit_cell_mapped[1];
							wq = w = -1 + (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								s = s - 0.5*unit_cell_mapped[0];
							else
								s = s + 0.5*unit_cell_mapped[0];

						}
						else if (sc == 5) {
							sq = s = interior_scale[0] * quadrature_abcissae[k];
							s = unit_cell_mapped[0] * (int(s / unit_cell_mapped[0]));
							tq = t = 1 - (i + 0.5)*unit_cell_mapped[1];
							wq = w = -1 + (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								s = s - 0.5*unit_cell_mapped[0];
							else
								s = s + 0.5*unit_cell_mapped[0];
						}
						else if (sc == 6) {
							sq = s = interior_scale[0] * quadrature_abcissae[k];
							s = unit_cell_mapped[0] * (int(s / unit_cell_mapped[0]));
							tq = t = -1 + (i + 0.5)*unit_cell_mapped[1];
							wq = w = 1 - (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								s = s - 0.5*unit_cell_mapped[0];
							else
								s = s + 0.5*unit_cell_mapped[0];
						}
						else if (sc == 7) {
							sq = s = interior_scale[0] * quadrature_abcissae[k];
							s = unit_cell_mapped[0] * (int(s / unit_cell_mapped[0]));
							tq = t = 1 - (i + 0.5)*unit_cell_mapped[1];
							wq = w = 1 - (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								s = s - 0.5*unit_cell_mapped[0];
							else
								s = s + 0.5*unit_cell_mapped[0];
						}
						else if (sc == 8) {
							sq = s = -1 + (i + 0.5)*unit_cell_mapped[0];
							tq = t = interior_scale[1] * quadrature_abcissae[k];
							t = unit_cell_mapped[1] * (int(t / unit_cell_mapped[1]));
							wq = w = -1 + (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								t = t - 0.5*unit_cell_mapped[1];
							else
								t = t + 0.5*unit_cell_mapped[1];

						}
						else if (sc == 9) {
							sq = s = 1 - (i + 0.5)*unit_cell_mapped[0];
							tq = t = interior_scale[1] * quadrature_abcissae[k];
							t = unit_cell_mapped[1] * (int(t / unit_cell_mapped[1]));
							wq = w = -1 + (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								t = t - 0.5*unit_cell_mapped[1];
							else
								t = t + 0.5*unit_cell_mapped[1];
						}
						else if (sc == 10) {
							sq = s = -1 + (i + 0.5)*unit_cell_mapped[0];
							tq = t = interior_scale[1] * quadrature_abcissae[k];
							t = unit_cell_mapped[1] * (int(t / unit_cell_mapped[1]));
							wq = w = 1 - (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								t = t - 0.5*unit_cell_mapped[1];
							else
								t = t + 0.5*unit_cell_mapped[1];
						}
						else if (sc == 11) {
							sq = s = 1 - (i + 0.5)*unit_cell_mapped[0];
							tq = t = interior_scale[1] * quadrature_abcissae[k];
							t = unit_cell_mapped[1] * (int(t / unit_cell_mapped[1]));
							wq = w = 1 - (j + 0.5)*unit_cell_mapped[2];
							if (quadrature_abcissae[k] < 0)
								t = t - 0.5*unit_cell_mapped[1];
							else
								t = t + 0.5*unit_cell_mapped[1];
						}



						quad_position[0] = 0;
						quad_position[1] = 0;
						quad_position[2] = 0;
						for (int kkk = 0; kkk < nodes_per_element; kkk++) {
							shape_func = shape_function(s, t, w, 2, kkk + 1);
							quad_position[0] += current_nodal_positions[kkk][poly_counter][0] * shape_func;
							quad_position[1] += current_nodal_positions[kkk][poly_counter][1] * shape_func;
							quad_position[2] += current_nodal_positions[kkk][poly_counter][2] * shape_func;
						}

						if(quad_position[0]>bounding_boxlo[0]&&quad_position[0]<bounding_boxhi[0]&&
					quad_position[1]>bounding_boxlo[1]&&quad_position[1]<bounding_boxhi[1]&&
					quad_position[2]>bounding_boxlo[2]&&quad_position[2]<bounding_boxhi[2]) {
						found_flag = 1;
						return found_flag;
					}
					}

				}

			}
		}
	


	//compute corner contributions

	for (int sc = 0; sc < 8; sc++) {
		for (int i = 0; i < surface_counts[0]; i++) {//alter surface counts for specific corner
			for (int j = 0; j < surface_counts[1]; j++) {
				for (int k = 0; k < surface_counts[2]; k++) {
					
					
						if (sc == 0) {

							s = -1 + (i + 0.5)*unit_cell_mapped[0];
							t = -1 + (j + 0.5)*unit_cell_mapped[1];
							w = -1 + (k + 0.5)*unit_cell_mapped[2];

						}
						else if (sc == 1) {
							s = 1 - (i + 0.5)*unit_cell_mapped[0];
							t = -1 + (j + 0.5)*unit_cell_mapped[1];
							w = -1 + (k + 0.5)*unit_cell_mapped[2];
						}
						else if (sc == 2) {
							s = 1 - (i + 0.5)*unit_cell_mapped[0];
							t = 1 - (j + 0.5)*unit_cell_mapped[1];
							w = -1 + (k + 0.5)*unit_cell_mapped[2];
						}
						else if (sc == 3) {
							s = -1 + (i + 0.5)*unit_cell_mapped[0];
							t = 1 - (j + 0.5)*unit_cell_mapped[1];
							w = -1 + (k + 0.5)*unit_cell_mapped[2];
						}
						else if (sc == 4) {
							s = -1 + (i + 0.5)*unit_cell_mapped[0];
							t = -1 + (j + 0.5)*unit_cell_mapped[1];
							w = 1 - (k + 0.5)*unit_cell_mapped[2];

						}
						else if (sc == 5) {
							s = 1 - (i + 0.5)*unit_cell_mapped[0];
							t = -1 + (j + 0.5)*unit_cell_mapped[1];
							w = 1 - (k + 0.5)*unit_cell_mapped[2];
						}
						else if (sc == 6) {
							s = 1 - (i + 0.5)*unit_cell_mapped[0];
							t = 1 - (j + 0.5)*unit_cell_mapped[1];
							w = 1 - (k + 0.5)*unit_cell_mapped[2];
						}
						else if (sc == 7) {
							s = -1 + (i + 0.5)*unit_cell_mapped[0];
							t = 1 - (j + 0.5)*unit_cell_mapped[1];
							w = 1 - (k + 0.5)*unit_cell_mapped[2];
						}

						quad_position[0] = 0;
						quad_position[1] = 0;
						quad_position[2] = 0;
						for (int kkk = 0; kkk < nodes_per_element; kkk++) {
							shape_func = shape_function(s, t, w, 2, kkk + 1);
							quad_position[0] += current_nodal_positions[kkk][poly_counter][0] * shape_func;
							quad_position[1] += current_nodal_positions[kkk][poly_counter][1] * shape_func;
							quad_position[2] += current_nodal_positions[kkk][poly_counter][2] * shape_func;
						}

					if(quad_position[0]>bounding_boxlo[0]&&quad_position[0]<bounding_boxhi[0]&&
					quad_position[1]>bounding_boxlo[1]&&quad_position[1]<bounding_boxhi[1]&&
					quad_position[2]>bounding_boxlo[2]&&quad_position[2]<bounding_boxhi[2]) {
						found_flag = 1;
						return found_flag;
					}

					}
				}

			}
		}
	}


	

		return found_flag;
	
	

}