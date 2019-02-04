/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef NPAIR_CLASS

NPairStyle(CAC,
           NPairCAC,
	 NP_BIN | NP_ATOMONLY |
	NP_NEWTON | NP_NEWTOFF | NP_ORTHO | NP_TRI | NP_CAC)

#else

#ifndef LMP_NPAIR_CAC_H
#define LMP_NPAIR_CAC_H

#include "npair.h"

namespace LAMMPS_NS {

class NPairCAC : public NPair {
 public:

  NPairCAC(class LAMMPS *);
  ~NPairCAC() {}
  void build(class NeighList *);
  double shape_function(double, double, double, int, int);
  void compute_surface_depths(double &x, double &y, double &z,
	  int &xb, int &yb, int &zb, int flag);
  int CAC_decide(int index_one, int index_two);
  int CAC_decide_atom2element(int index_one, int index_two);
  double ***current_nodal_positions;
  double CAC_cut;
  int   quadrature_node_count;
  int **element_scale;
  int current_element_scale[3];
  int current_poly_counter;
  double *quadrature_weights;
  double *quadrature_abcissae;
  void quadrature_init(int degree);
};

}

#endif
#endif

/* ERROR/WARNING messages:

*/
