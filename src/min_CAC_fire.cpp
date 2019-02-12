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
#include "min_CAC_fire.h"
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
#define DT_GROW 1.1
#define DT_SHRINK 0.5
#define ALPHA0 0.1
#define ALPHA_SHRINK 0.99
#define TMAX 10.0

/* ---------------------------------------------------------------------- */

CACMinFire::CACMinFire(LAMMPS *lmp) : Min(lmp) {}

/* ---------------------------------------------------------------------- */

void CACMinFire::init()
{
  Min::init();

  dt = update->dt;
  dtmax = TMAX * dt;
  alpha = ALPHA0;
  last_negative = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void CACMinFire::setup_style()
{
  double **v = atom->nodal_velocities[0][0];
  int nlocal = atom->maxpoly*atom->nodes_per_element * atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    v[i][0] = v[i][1] = v[i][2] = 0.0;
}

/* ----------------------------------------------------------------------
   set current vector lengths and pointers
   called after atoms have migrated
------------------------------------------------------------------------- */

void CACMinFire::reset_vectors()
{
  // atomic dof

  nvec = 3*atom->maxpoly*atom->nodes_per_element * atom->nlocal;
  if (nvec) xvec = atom->nodal_positions[0][0][0];
  if (nvec) fvec = atom->nodal_forces[0][0][0];
}

/* ---------------------------------------------------------------------- */

int CACMinFire::iterate(int maxiter)
{
  bigint ntimestep;
  double vmax,vdotf,vdotfall,vdotv,vdotvall,fdotf,fdotfall;
  double scale1,scale2;
  double dtvone,dtv,dtf,dtfm;
  int flag,flagall;

  int *element_type = atom->element_type;
  int *poly_count = atom->poly_count;
  int nodes_per_element;
  double ****nodal_positions=atom->nodal_positions;


  alpha_final = 0.0;

  for (int iter = 0; iter < maxiter; iter++) {

    if (timer->check_timeout(niter))
      return TIMEOUT;

    ntimestep = ++update->ntimestep;
    niter++;

    // vdotfall = v dot f

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

    // if (v dot f) > 0:
    // v = (1-alpha) v + alpha |v| Fhat
    // |v| = length of v, Fhat = unit f
    // if more than DELAYSTEP since v dot f was negative:
    // increase timestep and decrease alpha

    if (vdotfall > 0.0) {
      vdotv = 0.0;
      for (int i = 0; i < nlocal; i++)
        vdotv += v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2];
      MPI_Allreduce(&vdotv,&vdotvall,1,MPI_DOUBLE,MPI_SUM,world);

      // sum vdotv over replicas, if necessary
      // this communicator would be invalid for multiprocess replicas

      if (update->multireplica == 1) {
        vdotv = vdotvall;
        MPI_Allreduce(&vdotv,&vdotvall,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
      }

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

      scale1 = 1.0 - alpha;
      if (fdotfall == 0.0) scale2 = 0.0;
      else scale2 = alpha * sqrt(vdotvall/fdotfall);
      for (int i = 0; i < nlocal; i++) {
        v[i][0] = scale1*v[i][0] + scale2*f[i][0];
        v[i][1] = scale1*v[i][1] + scale2*f[i][1];
        v[i][2] = scale1*v[i][2] + scale2*f[i][2];
      }

      if (ntimestep - last_negative > DELAYSTEP) {
        dt = MIN(dt*DT_GROW,dtmax);
        alpha *= ALPHA_SHRINK;
      }

    // else (v dot f) <= 0:
    // decrease timestep, reset alpha, set v = 0

    } else {
      last_negative = ntimestep;
      dt *= DT_SHRINK;
      alpha = ALPHA0;
      for (int i = 0; i < nlocal; i++)
        v[i][0] = v[i][1] = v[i][2] = 0.0;
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

    double **xx = atom->x;
    double **x = atom->nodal_positions[0][0];
    
    if (rmass) {
      for (int i = 0; i < nlocal; i++) {
        dtfm = dtf / rmass[i];
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
    } else {
      for (int i = 0; i < nlocal; i++) {
        dtfm = dtf / mass[type[i]];
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
    }
    // update x for elements and atoms using nodal variables
    for (int i = 0; i < atom->nlocal; i++){
      //determine element type
      if(element_type[i] == 0) {
        nodes_per_element = 1;
      }
      if(element_type[i] == 1) {
        nodes_per_element = 8;
      }
    
      xx[i][0] = 0;
      xx[i][1] = 0;
      xx[i][2] = 0;

      for(int k=0; k<nodes_per_element; k++){
        for (int poly_counter = 0; poly_counter < poly_count[i];poly_counter++) {
          
            xx[i][0] += nodal_positions[i][k][poly_counter][0];
            xx[i][1] += nodal_positions[i][k][poly_counter][1];
            xx[i][2] += nodal_positions[i][k][poly_counter][2];
          }
      }

      xx[i][0] = xx[i][0] / nodes_per_element / poly_count[i];
      xx[i][1] = xx[i][1] / nodes_per_element / poly_count[i];
      xx[i][2] = xx[i][2] / nodes_per_element / poly_count[i];
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
