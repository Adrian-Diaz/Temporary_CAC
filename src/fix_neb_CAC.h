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

#ifdef FIX_CLASS

FixStyle(CAC/neb,FixNEBCAC)

#else

#ifndef LMP_FIX_NEB_CAC_H
#define LMP_FIX_NEB_CAC_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNEBCAC : public Fix {
 public:
  double veng,plen,nlen,dotpath,dottangrad,gradlen,dotgrad;
  double plennode, nlennode, dotpathnode, dottangradnode, gradlennode, dotgradnode;
  int rclimber;

  FixNEBCAC(class LAMMPS *, int, char **);
  ~FixNEBCAC();
  int setmask();
  void init();
  void min_setup(int);
  void min_post_force(int);

 private:
  int me,nprocs,nprocs_universe;
  double kspring,kspringIni,kspringFinal,kspringPerp,EIniIni,EFinalIni;
  bool StandardNEB,NEBLongRange,PerpSpring,FreeEndIni,FreeEndFinal;
  bool FreeEndFinalWithRespToEIni,FinalAndInterWithRespToEIni;
  int ireplica,nreplica;
  int procnext,procprev;
  int cmode;
  MPI_Comm uworld;
  MPI_Comm rootworld;


  char *id_pe;
  class Compute *pe;

  int nebatoms;
  int ntotal;                  // total # of atoms, NEB or not
  int maxlocal;                // size of xprev,xnext,tangent arrays
  double *nlenall;
  double **xprev,**xnext,**fnext,**springF;
  double **tangent;
  double **xsend,**xrecv;      // coords to send/recv to/from other replica
  double **fsend,**frecv;      // coords to send/recv to/from other replica
  tagint *tagsend,*tagrecv;    // ditto for atom IDs

                                 // info gathered from all procs in my replica
  double **xsendall,**xrecvall;    // coords to send/recv to/from other replica
  double **fsendall,**frecvall;    // force to send/recv to/from other replica
  tagint *tagsendall,*tagrecvall;  // ditto for atom IDs


  //Send/receive for CAC elements
  int nebnode;
  int ntotalnode;                  // total # of atoms, NEB or not
  int maxlocalnode;                // size of xprev,xnext,tangent arrays
  double *nlenallnode;
  double ****xprevnode,****xnextnode,****fnextnode,****springFnode;
  double ****tangentnode;
  double ****xsendnode,****xrecvnode;      // coords to send/recv to/from other replica
  double ****fsendnode,****frecvnode;      // coords to send/recv to/from other replica
  tagint *tagsendnode,*tagrecvnode;    // ditto for atom IDs

                                 // info gathered from all procs in my replica
  double ****xsendallnode,****xrecvallnode;    // coords to send/recv to/from other replica
  double ****fsendallnode,****frecvallnode;    // force to send/recv to/from other replica
  tagint *tagsendallnode,*tagrecvallnode;  // ditto for atom IDs

  int *counts,*displacements;   // used for MPI_Gather

  void inter_replica_comm();
  void reallocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Potential energy ID for fix neb does not exist

Self-explanatory.

E: Atom count changed in fix neb

This is not allowed in a NEB calculation.

*/
