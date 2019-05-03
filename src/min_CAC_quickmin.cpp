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

#include <mpi.h>
#include <math.h>
#include "min_CAC_quickmin.h"
#include "universe.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "output.h"
#include "timer.h"
#include "error.h"

using namespace LAMMPS_NS;

// EPS_ENERGY = minimum normalization for energy tolerance

#define EPS_ENERGY 1.0e-8

#define DELAYSTEP 5

/* ---------------------------------------------------------------------- */

CACMinQuickMin::CACMinQuickMin(LAMMPS *lmp) : Min(lmp) {}

/* ---------------------------------------------------------------------- */

void CACMinQuickMin::init()
{
  Min::init();

  dt = update->dt;
  last_negative = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void CACMinQuickMin::setup_style()
{
  double **v = atom->nodal_velocities[0][0];
  double **atomv = atom->v;
  int nlocal = atom->maxpoly*atom->nodes_per_element * atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    v[i][0] = v[i][1] = v[i][2] = 0.0;
  for (int j = 0; j < atom->nlocal; j++)
    atomv[j][0] = atomv[j][1] = atomv[j][2] = 0.0;
}

/* ----------------------------------------------------------------------
   set current vector lengths and pointers
   called after atoms have migrated
------------------------------------------------------------------------- */

void CACMinQuickMin::reset_vectors()
{
  // atomic dof

  // nvec = 3 * atom->nlocal;
  // if (nvec) xvec = atom->x[0];
  // if (nvec) fvec = atom->f[0];

  nvec = 3*atom->maxpoly*atom->nodes_per_element * atom->nlocal;
  if (nvec) xvec = atom->nodal_positions[0][0][0];
  if (nvec) fvec = atom->nodal_forces[0][0][0];

}

/* ----------------------------------------------------------------------
   minimization via QuickMin damped dynamics
------------------------------------------------------------------------- */

int CACMinQuickMin::iterate(int maxiter)
{
  bigint ntimestep;
  double vmax,vdotf,vdotfall,fdotf,fdotfall,scale;
  double dtvone,dtv,dtf,dtfm;
  int flag,flagall;

  alpha_final = 0.0;

  for (int iter = 0; iter < maxiter; iter++) {

    if (timer->check_timeout(niter))
      return TIMEOUT;

    ntimestep = ++update->ntimestep;
    niter++;

    // zero velocity if anti-parallel to force
    // else project velocity in direction of force
    double **v = atom->nodal_velocities[0][0];
    double **f = atom->nodal_forces[0][0];
    int nlocal = atom->maxpoly*atom->nodes_per_element * atom->nlocal;

    vdotf = 0.0;
    for (int i = 0; i < nlocal; i++)
      vdotf += v[i][0]*f[i][0] + v[i][1]*f[i][1] + v[i][2]*f[i][2];
    MPI_Allreduce(&vdotf,&vdotfall,1,MPI_DOUBLE,MPI_SUM,world);

    // sum vdotf over replicas, if necessary
    // this communicator would be invalid for multiprocess replicas

    if (update->multireplica == 1) {
      vdotf = vdotfall;
      MPI_Allreduce(&vdotf,&vdotfall,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
    }

    if (vdotfall < 0.0) {
      last_negative = ntimestep;
      for (int i = 0; i < nlocal; i++)
        v[i][0] = v[i][1] = v[i][2] = 0.0;

    } else {
      fdotf = 0.0;
      for (int i = 0; i < nlocal; i++)
        fdotf += f[i][0]*f[i][0] + f[i][1]*f[i][1] + f[i][2]*f[i][2];
      MPI_Allreduce(&fdotf,&fdotfall,1,MPI_DOUBLE,MPI_SUM,world);

      // sum fdotf over replicas, if necessary
      // this communicator would be invalid for multiprocess replicas

      if (update->multireplica == 1) {
        fdotf = fdotfall;
        MPI_Allreduce(&fdotf,&fdotfall,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
      }

      if (fdotfall == 0.0) scale = 0.0;
      else scale = vdotfall/fdotfall;
      for (int i = 0; i < nlocal; i++) {
        v[i][0] = scale * f[i][0];
        v[i][1] = scale * f[i][1];
        v[i][2] = scale * f[i][2];
      }
    }

    // limit timestep so no particle moves further than dmax

    double *rmass = atom->rmass;
    double *mass = atom->mass;
    int *type = atom->type;

    dtvone = dt;

    for (int i = 0; i < nlocal; i++) {
      vmax = MAX(fabs(v[i][0]),fabs(v[i][1]));
      vmax = MAX(vmax,fabs(v[i][2]));
      if (dtvone*vmax > dmax) dtvone = dmax/vmax;
    }
    MPI_Allreduce(&dtvone,&dtv,1,MPI_DOUBLE,MPI_MIN,world);

    // min dtv over replicas, if necessary
    // this communicator would be invalid for multiprocess replicas

    if (update->multireplica == 1) {
      dtvone = dtv;
      MPI_Allreduce(&dtvone,&dtv,1,MPI_DOUBLE,MPI_MIN,universe->uworld);
    }

    dtf = dtv * force->ftm2v;

    // Euler integration step

    double ****xnode = atom->nodal_positions;
    double ****vnode = atom->nodal_velocities;
    double ****fnode = atom->nodal_forces;
    int *element_type = atom->element_type;
    int *poly_count = atom->poly_count;
    int **node_types = atom->node_types;
    int *nodes_count_list = atom->nodes_per_element_list;
    int nodes_per_element;
    // Use verbose looping due to nodal masses
    if (rmass) {
      for (int i = 0; i < atom->nlocal; i++) {
        nodes_per_element = nodes_count_list[element_type[i]];

        for (int j = 0; j < nodes_per_element; j++){
          for (int k = 0; k < poly_count[i]; k++){
            dtfm = dtf / rmass[i];
            xnode[i][j][k][0] += dtv * vnode[i][j][k][0];
            xnode[i][j][k][1] += dtv * vnode[i][j][k][1];
            xnode[i][j][k][2] += dtv * vnode[i][j][k][2];
            vnode[i][j][k][0] += dtfm * fnode[i][j][k][0];
            vnode[i][j][k][1] += dtfm * fnode[i][j][k][1];
            vnode[i][j][k][2] += dtfm * fnode[i][j][k][2];
          }
        }
      }
    } else {
      for (int i = 0; i < nlocal; i++) {
        nodes_per_element = nodes_count_list[element_type[i]];
        for (int j = 0; j < nodes_per_element; j++){
          for (int k = 0; k < poly_count[i]; k++){
            dtfm = dtf / mass[node_types[i][k]];
            xnode[i][j][k][0] += dtv * vnode[i][j][k][0];
            xnode[i][j][k][1] += dtv * vnode[i][j][k][1];
            xnode[i][j][k][2] += dtv * vnode[i][j][k][2];
            vnode[i][j][k][0] += dtfm * fnode[i][j][k][0];
            vnode[i][j][k][1] += dtfm * fnode[i][j][k][1];
            vnode[i][j][k][2] += dtfm * fnode[i][j][k][2];
          }
        }
      }
    }

    // update x for elements and atoms using nodal variables
    double **xatom = atom->x;
    double **vatom = atom->v;


    for (int i = 0; i < atom->nlocal; i++){
      //determine element type
      nodes_per_element = nodes_count_list[element_type[i]];    
      xatom[i][0] = xatom[i][1] = xatom[i][2] = 0;
      vatom[i][0] = vatom[i][1] = vatom[i][2] = 0;
      for(int k=0; k<nodes_per_element; k++){
        for (int poly_counter = 0; poly_counter < poly_count[i];poly_counter++) {
          xatom[i][0] += xnode[i][k][poly_counter][0];
          xatom[i][1] += xnode[i][k][poly_counter][1];
          xatom[i][2] += xnode[i][k][poly_counter][2];
          vatom[i][0] += vnode[i][k][poly_counter][0];
          vatom[i][1] += vnode[i][k][poly_counter][1];
          vatom[i][2] += vnode[i][k][poly_counter][2];
        }
      }
      xatom[i][0] = xatom[i][0] / nodes_per_element / poly_count[i];
      xatom[i][1] = xatom[i][1] / nodes_per_element / poly_count[i];
      xatom[i][2] = xatom[i][2] / nodes_per_element / poly_count[i];
      vatom[i][0] = vatom[i][0] / nodes_per_element / poly_count[i];
      vatom[i][1] = vatom[i][1] / nodes_per_element / poly_count[i];
      vatom[i][2] = vatom[i][2] / nodes_per_element / poly_count[i];
    }

    eprevious = ecurrent;
    ecurrent = energy_force(0);
    neval++;

    // energy tolerance criterion
    // only check after DELAYSTEP elapsed since velocties reset to 0
    // sync across replicas if running multi-replica minimization

    if (update->etol > 0.0 && ntimestep-last_negative > DELAYSTEP) {
      if (update->multireplica == 0) {
        if (fabs(ecurrent-eprevious) <
            update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
          return ETOL;
      } else {
        if (fabs(ecurrent-eprevious) <
            update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
          flag = 0;
        else flag = 1;
        MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,universe->uworld);
        if (flagall == 0) return ETOL;
      }
    }

    // force tolerance criterion
    // sync across replicas if running multi-replica minimization

    if (update->ftol > 0.0) {
      fdotf = fnorm_sqr();
      if (update->multireplica == 0) {
        if (fdotf < update->ftol*update->ftol) return FTOL;
      } else {
        if (fdotf < update->ftol*update->ftol) flag = 0;
        else flag = 1;
        MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,universe->uworld);
        if (flagall == 0) return FTOL;
      }
    }

    // output for thermo, dump, restart files

    if (output->next == ntimestep) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }

  return MAXITER;
}
