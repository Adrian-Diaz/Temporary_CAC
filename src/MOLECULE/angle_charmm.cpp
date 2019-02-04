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

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include "angle_charmm.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.001

/* ---------------------------------------------------------------------- */

AngleCharmm::AngleCharmm(LAMMPS *lmp) : Angle(lmp) {}

/* ---------------------------------------------------------------------- */

AngleCharmm::~AngleCharmm()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(theta0);
    memory->destroy(k_ub);
    memory->destroy(r_ub);
  }
}

/* ---------------------------------------------------------------------- */

void AngleCharmm::compute(int eflag, int vflag)
{
  int i1,i2,i3,n,type;
  double delx1,dely1,delz1,delx2,dely2,delz2;
  double eangle,f1[3],f3[3];
  double dtheta,tk;
  double rsq1,rsq2,r1,r2,c,s,a,a11,a12,a22;
  double delxUB,delyUB,delzUB,rsqUB,rUB,dr,rk,forceUB;

  eangle = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nanglelist; n++) {
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    type = anglelist[n][3];

    // 1st bond

    delx1 = x[i1][0] - x[i2][0];
    dely1 = x[i1][1] - x[i2][1];
    delz1 = x[i1][2] - x[i2][2];

    rsq1 = delx1*delx1 + dely1*dely1 + delz1*delz1;
    r1 = sqrt(rsq1);

    // 2nd bond

    delx2 = x[i3][0] - x[i2][0];
    dely2 = x[i3][1] - x[i2][1];
    delz2 = x[i3][2] - x[i2][2];

    rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
    r2 = sqrt(rsq2);

    // Urey-Bradley bond

    delxUB = x[i3][0] - x[i1][0];
    delyUB = x[i3][1] - x[i1][1];
    delzUB = x[i3][2] - x[i1][2];

    rsqUB = delxUB*delxUB + delyUB*delyUB + delzUB*delzUB;
    rUB = sqrt(rsqUB);

    // Urey-Bradley force & energy

    dr = rUB - r_ub[type];
    rk = k_ub[type] * dr;

    if (rUB > 0.0) forceUB = -2.0*rk/rUB;
    else forceUB = 0.0;

    if (eflag) eangle = rk*dr;

    // angle (cos and sin)

    c = delx1*delx2 + dely1*dely2 + delz1*delz2;
    c /= r1*r2;

    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;

    s = sqrt(1.0 - c*c);
    if (s < SMALL) s = SMALL;
    s = 1.0/s;

    // harmonic force & energy

    dtheta = acos(c) - theta0[type];
    tk = k[type] * dtheta;

    if (eflag) eangle += tk*dtheta;

    a = -2.0 * tk * s;
    a11 = a*c / rsq1;
    a12 = -a / (r1*r2);
    a22 = a*c / rsq2;

    f1[0] = a11*delx1 + a12*delx2 - delxUB*forceUB;
    f1[1] = a11*dely1 + a12*dely2 - delyUB*forceUB;
    f1[2] = a11*delz1 + a12*delz2 - delzUB*forceUB;

    f3[0] = a22*delx2 + a12*delx1 + delxUB*forceUB;
    f3[1] = a22*dely2 + a12*dely1 + delyUB*forceUB;
    f3[2] = a22*delz2 + a12*delz1 + delzUB*forceUB;

    // apply force to each of 3 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += f1[0];
      f[i1][1] += f1[1];
      f[i1][2] += f1[2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= f1[0] + f3[0];
      f[i2][1] -= f1[1] + f3[1];
      f[i2][2] -= f1[2] + f3[2];
    }

    if (newton_bond || i3 < nlocal) {
      f[i3][0] += f3[0];
      f[i3][1] += f3[1];
      f[i3][2] += f3[2];
    }

    if (evflag) ev_tally(i1,i2,i3,nlocal,newton_bond,eangle,f1,f3,
                         delx1,dely1,delz1,delx2,dely2,delz2);
  }
}

/* ---------------------------------------------------------------------- */

void AngleCharmm::allocate()
{
  allocated = 1;
  int n = atom->nangletypes;

  memory->create(k,n+1,"angle:k");
  memory->create(theta0,n+1,"angle:theta0");
  memory->create(k_ub,n+1,"angle:k_ub");
  memory->create(r_ub,n+1,"angle:r_ub");
  memory->create(setflag,n+1,"angle:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

void AngleCharmm::coeff(int narg, char **arg)
{
  if (narg != 5) error->all(FLERR,"Incorrect args for angle coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->nangletypes,ilo,ihi);

  double k_one = force->numeric(FLERR,arg[1]);
  double theta0_one = force->numeric(FLERR,arg[2]);
  double k_ub_one = force->numeric(FLERR,arg[3]);
  double r_ub_one = force->numeric(FLERR,arg[4]);

  // convert theta0 from degrees to radians

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    theta0[i] = theta0_one/180.0 * MY_PI;
    k_ub[i] = k_ub_one;
    r_ub[i] = r_ub_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for angle coefficients");
}

/* ---------------------------------------------------------------------- */

double AngleCharmm::equilibrium_angle(int i)
{
  return theta0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void AngleCharmm::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&theta0[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&k_ub[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&r_ub[1],sizeof(double),atom->nangletypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void AngleCharmm::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&k[1],sizeof(double),atom->nangletypes,fp);
    fread(&theta0[1],sizeof(double),atom->nangletypes,fp);
    fread(&k_ub[1],sizeof(double),atom->nangletypes,fp);
    fread(&r_ub[1],sizeof(double),atom->nangletypes,fp);
  }
  MPI_Bcast(&k[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&theta0[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&k_ub[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r_ub[1],atom->nangletypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nangletypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void AngleCharmm::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nangletypes; i++)
    fprintf(fp,"%d %g %g %g %g\n",
            i,k[i],theta0[i]/MY_PI*180.0,k_ub[i],r_ub[i]);
}

/* ---------------------------------------------------------------------- */

double AngleCharmm::single(int type, int i1, int i2, int i3)
{
  double **x = atom->x;

  double delx1 = x[i1][0] - x[i2][0];
  double dely1 = x[i1][1] - x[i2][1];
  double delz1 = x[i1][2] - x[i2][2];
  domain->minimum_image(delx1,dely1,delz1);
  double r1 = sqrt(delx1*delx1 + dely1*dely1 + delz1*delz1);

  double delx2 = x[i3][0] - x[i2][0];
  double dely2 = x[i3][1] - x[i2][1];
  double delz2 = x[i3][2] - x[i2][2];
  domain->minimum_image(delx2,dely2,delz2);
  double r2 = sqrt(delx2*delx2 + dely2*dely2 + delz2*delz2);

  double delxUB = x[i3][0] - x[i1][0];
  double delyUB = x[i3][1] - x[i1][1];
  double delzUB = x[i3][2] - x[i1][2];
  domain->minimum_image(delxUB,delyUB,delzUB);
  double rUB = sqrt(delxUB*delxUB + delyUB*delyUB + delzUB*delzUB);

  double c = delx1*delx2 + dely1*dely2 + delz1*delz2;
  c /= r1*r2;
  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;

  double dtheta = acos(c) - theta0[type];
  double tk = k[type] * dtheta;
  double dr = rUB - r_ub[type];
  double rk = k_ub[type] * dr;

  return (tk*dtheta + rk*dr);
}
