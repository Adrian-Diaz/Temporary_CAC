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
   Contributing author for: Emile Maras (CEA, France)
     new options for inter-replica forces, first/last replica treatment
------------------------------------------------------------------------- */

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fix_neb_CAC.h"
#include "universe.h"
#include "update.h"
#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "modify.h"
#include "compute.h"
#include "group.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{SINGLE_PROC_DIRECT,SINGLE_PROC_MAP,MULTI_PROC};

#define BUFSIZE 8

/* ---------------------------------------------------------------------- */

FixNEBCAC::FixNEBCAC(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  id_pe(NULL), pe(NULL), nlenall(NULL), xprev(NULL), xnext(NULL),
  fnext(NULL), springF(NULL), tangent(NULL), xsend(NULL), xrecv(NULL),
  fsend(NULL), frecv(NULL), tagsend(NULL), tagrecv(NULL),
  xsendall(NULL), xrecvall(NULL), fsendall(NULL), frecvall(NULL),
  tagsendall(NULL), tagrecvall(NULL), counts(NULL),
  displacements(NULL), nlenallnode(NULL), xprevnode(NULL), xnextnode(NULL),
  fnextnode(NULL), springFnode(NULL), tangentnode(NULL), xsendnode(NULL), xrecvnode(NULL),
  fsendnode(NULL), frecvnode(NULL), tagsendnode(NULL), tagrecvnode(NULL),
  xsendallnode(NULL), xrecvallnode(NULL), fsendallnode(NULL), frecvallnode(NULL),
  tagsendallnode(NULL), tagrecvallnode(NULL)
{
  if (narg < 4) error->all(FLERR,"Illegal fix neb command");

  kspring = force->numeric(FLERR,arg[3]);
  if (kspring <= 0.0) error->all(FLERR,"Illegal fix neb command");

  // optional params

  NEBLongRange = false;
  StandardNEB = true;
  PerpSpring = FreeEndIni = FreeEndFinal = false;
  FreeEndFinalWithRespToEIni = FinalAndInterWithRespToEIni = false;
  kspringPerp = 0.0;
  kspringIni = 1.0;
  kspringFinal = 1.0;

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"parallel") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix neb command");
      if (strcmp(arg[iarg+1],"ideal") == 0) {
        NEBLongRange = true;
        StandardNEB = false;
      } else if (strcmp(arg[iarg+1],"neigh") == 0) {
        NEBLongRange = false;
        StandardNEB = true;
      } else error->all(FLERR,"Illegal fix neb command");
      iarg += 2;

    } else if (strcmp(arg[iarg],"perp") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix neb command");
      PerpSpring = true;
      kspringPerp = force->numeric(FLERR,arg[iarg+1]);
      if (kspringPerp == 0.0) PerpSpring = false;
      if (kspringPerp < 0.0) error->all(FLERR,"Illegal fix neb command");
      iarg += 2;

    } else if (strcmp (arg[iarg],"end") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix neb command");
      if (strcmp(arg[iarg+1],"first") == 0) {
        FreeEndIni = true;
        kspringIni = force->numeric(FLERR,arg[iarg+2]);
      } else if (strcmp(arg[iarg+1],"last") == 0) {
        FreeEndFinal = true;
        FinalAndInterWithRespToEIni = false;
        FreeEndFinalWithRespToEIni = false;
        kspringFinal = force->numeric(FLERR,arg[iarg+2]);
      } else if (strcmp(arg[iarg+1],"last/efirst") == 0) {
        FreeEndFinal = false;
        FinalAndInterWithRespToEIni = false;
        FreeEndFinalWithRespToEIni = true;
        kspringFinal = force->numeric(FLERR,arg[iarg+2]);
      } else if (strcmp(arg[iarg+1],"last/efirst/middle") == 0) {
        FreeEndFinal = false;
        FinalAndInterWithRespToEIni = true;
        FreeEndFinalWithRespToEIni = true;
        kspringFinal = force->numeric(FLERR,arg[iarg+2]);
      } else error->all(FLERR,"Illegal fix neb command");

      iarg += 3;

    } else error->all(FLERR,"Illegal fix neb command");
  }

  // nreplica = number of partitions
  // ireplica = which world I am in universe
  // nprocs_universe = # of procs in all replicase
  // procprev,procnext = root proc in adjacent replicas

  me = comm->me;
  nprocs = comm->nprocs;

  nprocs_universe = universe->nprocs;
  nreplica = universe->nworlds;
  ireplica = universe->iworld;

  if (ireplica > 0) procprev = universe->root_proc[ireplica-1];
  else procprev = -1;
  if (ireplica < nreplica-1) procnext = universe->root_proc[ireplica+1];
  else procnext = -1;

  uworld = universe->uworld;
  int *iroots = new int[nreplica];
  MPI_Group uworldgroup,rootgroup;
  if (NEBLongRange) {
    for (int i=0; i<nreplica; i++)
      iroots[i] = universe->root_proc[i];
    MPI_Comm_group(uworld, &uworldgroup);
    MPI_Group_incl(uworldgroup, nreplica, iroots, &rootgroup);
    MPI_Comm_create(uworld, rootgroup, &rootworld);
  }
  delete [] iroots;


  // create a new compute pe style
  // id = fix-ID + pe, compute group = all

  int n = strlen(id) + 4;
  id_pe = new char[n];
  strcpy(id_pe,id);
  strcat(id_pe,"_pe");

  char **newarg = new char*[3];
  newarg[0] = id_pe;
  newarg[1] = (char *) "all";
  newarg[2] = (char *) "pe";
  modify->add_compute(3,newarg);
  delete [] newarg;

  // initialize local storage
  maxlocal = -1;
  ntotal = -1;
}

/* ---------------------------------------------------------------------- */

FixNEBCAC::~FixNEBCAC()
{
  modify->delete_compute(id_pe);
  delete [] id_pe;

  memory->destroy(xprev);
  memory->destroy(xnext);
  memory->destroy(tangent);
  memory->destroy(fnext);
  memory->destroy(springF);
  memory->destroy(xsend);
  memory->destroy(xrecv);
  memory->destroy(fsend);
  memory->destroy(frecv);
  memory->destroy(tagsend);
  memory->destroy(tagrecv);

  memory->destroy(xsendall);
  memory->destroy(xrecvall);
  memory->destroy(fsendall);
  memory->destroy(frecvall);
  memory->destroy(tagsendall);
  memory->destroy(tagrecvall);


  memory->destroy(xprevnode);
  memory->destroy(xnextnode);
  memory->destroy(tangentnode);
  memory->destroy(fnextnode);
  memory->destroy(springFnode);
  memory->destroy(xsendnode);
  memory->destroy(xrecvnode);
  memory->destroy(fsendnode);
  memory->destroy(frecvnode);
  memory->destroy(tagsendnode);
  memory->destroy(tagrecvnode);

  memory->destroy(xsendallnode);
  memory->destroy(xrecvallnode);
  memory->destroy(fsendallnode);
  memory->destroy(frecvallnode);
  memory->destroy(tagsendallnode);
  memory->destroy(tagrecvallnode);

  memory->destroy(counts);
  memory->destroy(displacements);

  if (NEBLongRange) {
    if (rootworld != MPI_COMM_NULL) MPI_Comm_free(&rootworld);
    memory->destroy(nlenall);
    memory->destroy(nlenallnode);
  }
}

/* ---------------------------------------------------------------------- */

int FixNEBCAC::setmask()
{
  int mask = 0;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNEBCAC::init()
{

  int icompute = modify->find_compute(id_pe);
  if (icompute < 0)
    error->all(FLERR,"Potential energy ID for fix neb does not exist");
  pe = modify->compute[icompute];

  // turn off climbing mode, NEB command turns it on after init()

  rclimber = -1;

  // nebatoms = # of atoms in fix group = atoms with inter-replica forces

  bigint count = group->count(igroup);
  if (count > MAXSMALLINT) error->all(FLERR,"Too many active NEB atoms");
  nebatoms = count;

  // comm mode for inter-replica exchange of coords

  if (nreplica == nprocs_universe &&
      nebatoms == atom->natoms && atom->sortfreq == 0)
    cmode = SINGLE_PROC_DIRECT;
  else if (nreplica == nprocs_universe) cmode = SINGLE_PROC_MAP;
  else cmode = MULTI_PROC;

  // ntotal = total # of atoms in system, NEB atoms or not

  if (atom->natoms > MAXSMALLINT) error->all(FLERR,"Too many atoms for NEB");
  ntotal = atom->natoms;

  if (atom->nmax > maxlocal) reallocate();

  //TODO: MULTI_PROC mode requires changes here.
  if (MULTI_PROC && counts == NULL) {
    memory->create(xsendall,ntotal,3,"neb_CAC:xsendall");
    memory->create(xrecvall,ntotal,3,"neb_CAC:xrecvall");
    memory->create(fsendall,ntotal,3,"neb_CAC:fsendall");
    memory->create(frecvall,ntotal,3,"neb_CAC:frecvall"); 
    memory->create(tagsendall,ntotal,"neb_CAC:tagsendall");
    memory->create(tagrecvall,ntotal,"neb_CAC:tagrecvall");
    memory->create(counts,nprocs,"neb_CAC:counts");
    memory->create(displacements,nprocs,"neb_CAC:displacements");
    memory->create(xsendallnode, ntotal, atom-> nodes_per_element, atom->maxpoly,3, "neb_CAC:xsendallnode");
    memory->create(xrecvallnode, ntotal, atom-> nodes_per_element, atom->maxpoly,3, "neb_CAC:xrecvallnode");
    memory->create(fsendallnode, ntotal, atom-> nodes_per_element, atom->maxpoly,3, "neb_CAC:fsendallnode");
    memory->create(frecvallnode, ntotal, atom-> nodes_per_element, atom->maxpoly,3, "neb_CAC:frecvallnode");

  }
}

/* ---------------------------------------------------------------------- */

void FixNEBCAC::min_setup(int vflag)
{
  min_post_force(vflag);

  // trigger potential energy computation on next timestep

  pe->addstep(update->ntimestep+1);
}

/* ---------------------------------------------------------------------- */

void FixNEBCAC::min_post_force(int vflag)
{
  double vprev,vnext;
  double delxp,delyp,delzp,delxn,delyn,delzn;
  double vIni=0.0;


  vprev = vnext = veng = pe->compute_scalar();

  if (ireplica < nreplica-1 && me == 0)
    MPI_Send(&veng,1,MPI_DOUBLE,procnext,0,uworld);
  if (ireplica > 0 && me == 0)
    MPI_Recv(&vprev,1,MPI_DOUBLE,procprev,0,uworld,MPI_STATUS_IGNORE);

  if (ireplica > 0 && me == 0)
    MPI_Send(&veng,1,MPI_DOUBLE,procprev,0,uworld);
  if (ireplica < nreplica-1 && me == 0)
    MPI_Recv(&vnext,1,MPI_DOUBLE,procnext,0,uworld,MPI_STATUS_IGNORE);

  if (cmode == MULTI_PROC) {
    MPI_Bcast(&vprev,1,MPI_DOUBLE,0,world);
    MPI_Bcast(&vnext,1,MPI_DOUBLE,0,world);
  }

  if (FreeEndFinal && ireplica == nreplica-1 && (update->ntimestep == 0)) EFinalIni = veng;

  if (ireplica == 0) vIni=veng;

  if (FreeEndFinalWithRespToEIni) {
    if (cmode == SINGLE_PROC_DIRECT || cmode == SINGLE_PROC_MAP) {
      int procFirst;
      procFirst=universe->root_proc[0];
      MPI_Bcast(&vIni,1,MPI_DOUBLE,procFirst,uworld);
    }else {
      if (me == 0)
        MPI_Bcast(&vIni,1,MPI_DOUBLE,0,rootworld);

      MPI_Bcast(&vIni,1,MPI_DOUBLE,0,world);
    }
  }

  if (FreeEndIni && ireplica == 0 && (update->ntimestep == 0)) EIniIni = veng;
  /*  if (FreeEndIni && ireplica == 0) {
    //    if (me == 0 )
      if (update->ntimestep == 0) {
        EIniIni = veng;
        //      if (cmode == MULTI_PROC)
        // MPI_Bcast(&EIniIni,1,MPI_DOUBLE,0,world);
      }
      }*/

  // communicate atoms to/from adjacent replicas to fill xprev,xnext, xprevnode, xnextnode

  inter_replica_comm();

  // trigger potential energy computation on next timestep

  pe->addstep(update->ntimestep+1);

  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  double dotnode = 0.0;
  double prefactornode = 0.0;

  double ****xnode = atom->nodal_positions;
  double ****fnode = atom->nodal_forces;
  double ****xprevn = xprevnode;
  double ****xnextn = xnextnode;
  double ****fnextn = fnextnode;
  int *element_type = atom->element_type;
  int *poly_count = atom->poly_count;
  int **node_types = atom->node_types;
  int *nodes_count_list = atom->nodes_per_element_list; 

  int nodes_per_element;
  int nlocal = atom->nlocal;

  //calculating separation between images
  plennode = 0.0;
  nlennode = 0.0;
  double tlennode = 0.0;
  double gradnextlennode = 0.0;

  dotgrad = gradlen = dotpath = dottangrad = 0.0;
  dotgradnode = gradlennode = dotpathnode = dottangradnode = 0.0;

  if (ireplica == nreplica-1) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        nodes_per_element = nodes_count_list[element_type[i]];
        for (int p = 0; p < nodes_per_element; p++) {
          for (int k = 0; k < poly_count[i]; k++){
            delxp = xnode[i][p][k][0] - xprevn[i][p][k][0];
            delyp = xnode[i][p][k][1] - xprevn[i][p][k][1];
            delzp = xnode[i][p][k][2] - xprevn[i][p][k][2];
            domain->minimum_image(delxp,delyp,delzp);
            plennode += delxp*delxp + delyp*delyp + delzp*delzp;
            dottangradnode += delxp* fnode[i][p][k][0]
              + delyp*fnode[i][p][k][1]
              + delzp*fnode[i][p][k][2];
            gradlennode += fnode[i][p][k][0]*fnode[i][p][k][0]
              + fnode[i][p][k][1]*fnode[i][p][k][1]
              + fnode[i][p][k][2]*fnode[i][p][k][2];
            if (FreeEndFinal||FreeEndFinalWithRespToEIni) {
              tangentnode[i][p][k][0]=delxp;
              tangentnode[i][p][k][1]=delyp;
              tangentnode[i][p][k][2]=delzp;
              tlennode += tangentnode[i][p][k][0]*tangentnode[i][p][k][0] 
                + tangentnode[i][p][k][1]*tangentnode[i][p][k][1]
                + tangentnode[i][p][k][2]*tangentnode[i][p][k][2];
              dotnode += fnode[i][p][k][0]*tangentnode[i][p][k][0]
                + fnode[i][p][k][1]*tangentnode[i][p][k][1]
                + fnode[i][p][k][2]*tangentnode[i][p][k][2];
            }
          }
        }
      }
    }
  }

  else if (ireplica == 0) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        nodes_per_element = nodes_count_list[element_type[i]];
        for (int p = 0; p < nodes_per_element; p++) {
          for (int k = 0; k < poly_count[i]; k++){
            delxn = xnextn[i][p][k][0] - xnode[i][p][k][0];
            delyn = xnextn[i][p][k][1] - xnode[i][p][k][1];
            delzn = xnextn[i][p][k][2] - xnode[i][p][k][2];
            domain->minimum_image(delxn,delyn,delzn);
            nlennode += delxn*delxn + delyn*delyn + delzn*delzn;
            gradnextlen += fnextn[i][p][k][0]*fnextn[i][p][k][0] 
              + fnextn[i][p][k][1]*fnextn[i][p][k][1] 
              + fnextn[i][p][k][2] * fnextn[i][p][k][2];
            dotgradnode += fnode[i][p][k][0]*fnextn[i][p][k][0] 
              + fnode[i][p][k][1]*fnextn[i][p][k][1]
              + fnode[i][p][k][2]*fnextn[i][p][k][2];
            dottangradnode += delxn* fnode[i][p][k][0]
              + delyn*fnode[i][p][k][1]
              + delzn*fnode[i][p][k][2];
            gradlennode += fnode[i][p][k][0]*fnode[i][p][k][0]
              + fnode[i][p][k][1]*fnode[i][p][k][1]
              + fnode[i][p][k][2]*fnode[i][p][k][2];
            if (FreeEndFinal||FreeEndFinalWithRespToEIni) {
              tangentnode[i][p][k][0]=delxn;
              tangentnode[i][p][k][1]=delyn;
              tangentnode[i][p][k][2]=delzn;
              tlennode += tangentnode[i][p][k][0]*tangentnode[i][p][k][0] 
                + tangentnode[i][p][k][1]*tangentnode[i][p][k][1]
                + tangentnode[i][p][k][2]*tangentnode[i][p][k][2];
              dotnode += fnode[i][p][k][0]*tangentnode[i][p][k][0]
                + fnode[i][p][k][1]*tangentnode[i][p][k][1]
                + fnode[i][p][k][2]*tangentnode[i][p][k][2];
            }
          }
        }
      }
    }
  } 

  else {

    // not the first or last replica

    double vmax = MAX(fabs(vnext-veng),fabs(vprev-veng));
    double vmin = MIN(fabs(vnext-veng),fabs(vprev-veng));


    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        nodes_per_element = nodes_count_list[element_type[i]];
        for (int p = 0; p < nodes_per_element; p++) {
          for (int k = 0; k < poly_count[i]; k++){
            delxp = xnode[i][p][k][0] - xprevn[i][p][k][0];
            delyp = xnode[i][p][k][1] - xprevn[i][p][k][1];
            delzp = xnode[i][p][k][2] - xprevn[i][p][k][2];
            domain->minimum_image(delxp,delyp,delzp);
            plennode += delxp*delxp + delyp*delyp + delzp*delzp;

            delxn = xnextn[i][p][k][0] - xnode[i][p][k][0];
            delyn = xnextn[i][p][k][1] - xnode[i][p][k][1];
            delzn = xnextn[i][p][k][2] - xnode[i][p][k][2];
            domain->minimum_image(delxn,delyn,delzn);

            if (vnext > veng && veng > vprev) {
              tangentnode[i][p][k][0] = delxn;
              tangentnode[i][p][k][1] = delyn;
              tangentnode[i][p][k][2] = delzn;
            } else if (vnext < veng && veng < vprev) {
              tangentnode[i][p][k][0] = delxp;
              tangentnode[i][p][k][1] = delyp;
              tangentnode[i][p][k][2] = delzp;
            } else {
              if (vnext > vprev) {
                tangentnode[i][p][k][0] = vmax*delxn + vmin*delxp;
                tangentnode[i][p][k][1] = vmax*delyn + vmin*delyp;
                tangentnode[i][p][k][2] = vmax*delzn + vmin*delzp;
              } else if (vnext < vprev) {
                tangentnode[i][p][k][0] = vmin*delxn + vmax*delxp;
                tangentnode[i][p][k][1] = vmin*delyn + vmax*delyp;
                tangentnode[i][p][k][2] = vmin*delzn + vmax*delzp;
              } else { // vnext == vprev, e.g. for potentials that do not compute an energy
                tangentnode[i][p][k][0] = delxn + delxp;
                tangentnode[i][p][k][1] = delyn + delyp;
                tangentnode[i][p][k][2] = delzn + delzp;
              }
            }

            nlennode += delxn*delxn + delyn*delyn + delzn*delzn;
            tlennode += tangentnode[i][p][k][0]*tangentnode[i][p][k][0]
              + tangentnode[i][p][k][1]*tangentnode[i][p][k][1]
              + tangentnode[i][p][k][2]*tangentnode[i][p][k][2];
            gradlennode += fnode[i][p][k][0]*fnode[i][p][k][0]
              + fnode[i][p][k][1]*fnode[i][p][k][1]
              + fnode[i][p][k][2]*fnode[i][p][k][2];
            dotpathnode += delxp*delxn + delyp*delyn + delzp*delzn;
            dottangradnode += tangentnode[i][p][k][0]*fnode[i][p][k][0]
              + tangentnode[i][p][k][1]*fnode[i][p][k][1]
              + tangentnode[i][p][k][2]*fnode[i][p][k][2];
            gradnextlennode += fnextn[i][p][k][0]*fnextn[i][p][k][0] 
              + fnextn[i][p][k][1]*fnextn[i][p][k][1]
              + fnextn[i][p][k][2] * fnextn[i][p][k][2];
            dotgradnode += fnode[i][p][k][0]*fnextn[i][p][k][0]
              + fnode[i][p][k][1]*fnextn[i][p][k][1]
              + fnode[i][p][k][2]*fnextn[i][p][k][2];

            springFnode[i][p][k][0] = kspringPerp*(delxn-delxp);
            springFnode[i][p][k][1] = kspringPerp*(delyn-delyp);
            springFnode[i][p][k][2] = kspringPerp*(delzn-delzp);
          }
        }
      }
    }
  }

  double nbufin[BUFSIZE], nbufout[BUFSIZE];
  nbufin[0] = nlennode;
  nbufin[1] = plennode;
  nbufin[2] = tlennode;
  nbufin[3] = gradlennode;
  nbufin[4] = gradnextlennode;
  nbufin[5] = dotpathnode;
  nbufin[6] = dottangradnode;
  nbufin[7] = dotgradnode;

  MPI_Allreduce(nbufin,nbufout,BUFSIZE,MPI_DOUBLE,MPI_SUM,world);
  nlennode = sqrt(nbufout[0]);
  plennode = sqrt(nbufout[1]);
  tlennode = sqrt(nbufout[2]);
  gradlennode = sqrt(nbufout[3]);
  gradnextlennode = sqrt(nbufout[4]);
  dotpathnode = nbufout[5];
  dottangradnode = nbufout[6];
  dotgradnode = nbufout[7];

  // normalize tangent vectors

  if (tlennode > 0.0) {
    double tleninvnode = 1.0/tlennode;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        nodes_per_element = nodes_count_list[element_type[i]];
        for (int p = 0; p < nodes_per_element; p++) {
          for (int k = 0; k < poly_count[i]; k++){
            tangentnode[i][p][k][0] *= tleninvnode;
            tangentnode[i][p][k][1] *= tleninvnode;
            tangentnode[i][p][k][2] *= tleninvnode;
          }
        }
      }
    }
  }

  // first or last replica has no change to forces, just return

  if (ireplica > 0 && ireplica < nreplica-1)
    dottangradnode = dottangradnode/(tlennode*gradlennode);
  if (ireplica == 0)
    dottangradnode = dottangradnode/(nlennode*gradlennode);
  if (ireplica == nreplica-1)
    dottangradnode = dottangradnode/(plennode*gradlennode);
  if (ireplica < nreplica-1)
    dotgradnode = dotgradnode /(gradlennode*gradnextlennode);

  if (FreeEndIni && ireplica == 0) {
    if (tlennode > 0.0) { 
      double dotallnode;
      MPI_Allreduce(&dotnode,&dotallnode,1,MPI_DOUBLE,MPI_SUM,world);
      dotnode = dotallnode/tlennode;

      if (dotnode < 0) prefactornode = -dotnode - kspringIni*(veng-EIniIni);
      else prefactornode = -dotnode + kspringIni*(veng-EIniIni);

      for (int i = 0; i < nlocal; i++) { 
        nodes_per_element = nodes_count_list[element_type[i]];
        if (mask[i] & groupbit) {
          for (int p = 0; p < nodes_per_element; p++) {
            for (int k = 0; k < poly_count[i]; k++){
                fnode[i][p][k][0] += prefactornode *tangentnode[i][p][k][0];
                fnode[i][p][k][1] += prefactornode *tangentnode[i][p][k][1];
                fnode[i][p][k][2] += prefactornode *tangentnode[i][p][k][2];
            }
          }
        }
      }
    }
  }

  if (FreeEndFinal && ireplica == nreplica -1) {
    if (tlennode > 0.0) { 
      double dotallnode;
      MPI_Allreduce(&dotnode,&dotallnode,1,MPI_DOUBLE,MPI_SUM,world);
      dotnode = dotallnode/tlennode;

      if (dotnode < 0) prefactornode = -dotnode - kspringFinal*(veng-EFinalIni);
      else prefactornode = -dotnode + kspringFinal*(veng-EFinalIni);

      for (int i = 0; i < nlocal; i++) { 
        nodes_per_element = nodes_count_list[element_type[i]];
        if (mask[i] & groupbit) {
          for (int p = 0; p < nodes_per_element; p++) {
            for (int k = 0; k < poly_count[i]; k++){
                fnode[i][p][k][0] += prefactornode *tangentnode[i][p][k][0];
                fnode[i][p][k][1] += prefactornode *tangentnode[i][p][k][1];
                fnode[i][p][k][2] += prefactornode *tangentnode[i][p][k][2];
            }
          }
        }
      }
    }
  }

  if (FreeEndFinalWithRespToEIni&&ireplica == nreplica -1) {
    if (tlennode > 0.0) { 
      double dotallnode;
      MPI_Allreduce(&dotnode,&dotallnode,1,MPI_DOUBLE,MPI_SUM,world);
      dotnode = dotallnode/tlennode;
      if (veng < vIni) {
        if (dotnode < 0) prefactornode = -dotnode - kspringFinal*(veng-vIni);
        else prefactornode = -dotnode + kspringFinal*(veng-vIni);
      }

      for (int i = 0; i < nlocal; i++) { 
        nodes_per_element = nodes_count_list[element_type[i]];
        if (mask[i] & groupbit) {
          for (int p = 0; p < nodes_per_element; p++) {
            for (int k = 0; k < poly_count[i]; k++){
                fnode[i][p][k][0] += prefactornode *tangentnode[i][p][k][0];
                fnode[i][p][k][1] += prefactornode *tangentnode[i][p][k][1];
                fnode[i][p][k][2] += prefactornode *tangentnode[i][p][k][2];
            }
          }
        }
      }
    }
  }

  double lentot = 0;
  double meanDist,idealPos,lenuntilIm,lenuntilClimber;
  lenuntilClimber=0;
  if (NEBLongRange) {
    if (cmode == SINGLE_PROC_DIRECT || cmode == SINGLE_PROC_MAP) {
      MPI_Allgather(&nlennode,1,MPI_DOUBLE,&nlenallnode[0],1,MPI_DOUBLE,uworld);
    } else {
      if (me == 0)
        MPI_Allgather(&nlennode,1,MPI_DOUBLE,&nlenallnode[0],1,MPI_DOUBLE,rootworld);
      MPI_Bcast(nlenallnode,nreplica,MPI_DOUBLE,0,world);
    }

    lenuntilIm = 0;
    for (int i = 0; i < ireplica; i++)
      lenuntilIm += nlenallnode[i];

    for (int i = 0; i < nreplica; i++)
      lentot += nlenallnode[i];

    meanDist = lentot/(nreplica -1);

    if (rclimber>0) {
      for (int i = 0; i < rclimber; i++)
        lenuntilClimber += nlenallnode[i];
      double meanDistBeforeClimber = lenuntilClimber/rclimber;
      double meanDistAfterClimber =
        (lentot-lenuntilClimber)/(nreplica-rclimber-1);
      if (ireplica<rclimber)
        idealPos = ireplica * meanDistBeforeClimber;
      else
        idealPos = lenuntilClimber+ (ireplica-rclimber)*meanDistAfterClimber;
    } else idealPos = ireplica * meanDist;
  }

  if (ireplica == 0 || ireplica == nreplica-1) return ;

  double AngularContrN;
  dotpathnode = dotpathnode/(plennode*nlennode);
  AngularContrN = 0.5 *(1+cos(MY_PI * dotpathnode));


  double dotSpringTangentN = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      nodes_per_element = nodes_count_list[element_type[i]];
      for (int p = 0; p < nodes_per_element; p++) {
        for (int k = 0; k < poly_count[i]; k++){
          dotnode += fnode[i][p][k][0] * tangentnode[i][p][k][0]
            + fnode[i][p][k][1] * tangentnode[i][p][k][1]
            + fnode[i][p][k][2] * tangentnode[i][p][k][2];
          dotSpringTangentN += springFnode[i][p][k][0] * tangentnode[i][p][k][0]
            + springFnode[i][p][k][1] * tangentnode[i][p][k][1]
            + springFnode[i][p][k][2] * tangentnode[i][p][k][2];
        }
      }
    }
  }

  double dotSpringTangentNall;
  MPI_Allreduce(&dotSpringTangentN,&dotSpringTangentNall,1,
                MPI_DOUBLE,MPI_SUM,world);
  dotSpringTangentN = dotSpringTangentNall;

  double dotallN;
  MPI_Allreduce(&dotnode,&dotallN,1,MPI_DOUBLE,MPI_SUM,world);
  dotnode = dotallN;

  if (ireplica == rclimber)
    prefactornode = -2.0*dotnode;
  else {
    if (NEBLongRange)
      prefactornode = -dot - kspring*(lenuntilIm-idealPos)/(2*meanDist);
    else if (StandardNEB)
      prefactornode = -dotnode + kspring*(nlennode-plennode);

    if (FinalAndInterWithRespToEIni&& veng<vIni) {
      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          nodes_per_element = nodes_count_list[element_type[i]];
          for (int p = 0; p < nodes_per_element; p++) {
            for (int k = 0; k < poly_count[i]; k++){
              fnode[i][p][k][0] = 0;
              fnode[i][p][k][1] = 0;
              fnode[i][p][k][2] = 0;
            }
          }
        }
      }
      prefactornode = kspring*(nlennode - plennode);
      AngularContrN = 0;
    }
  }

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      nodes_per_element = nodes_count_list[element_type[i]];
      for (int p = 0; p < nodes_per_element; p++) {
        for (int k = 0; k < poly_count[i]; k++){
          fnode[i][p][k][0] += prefactornode*tangentnode[i][p][k][0]
            + AngularContrN*(springFnode[i][p][k][0] - dotSpringTangentN*tangentnode[i][p][k][0]);
          fnode[i][p][k][1] += prefactornode*tangentnode[i][p][k][1]
            + AngularContrN*(springFnode[i][p][k][1] - dotSpringTangentN*tangentnode[i][p][k][1]);
          fnode[i][p][k][2] += prefactornode*tangentnode[i][p][k][2]
            + AngularContrN*(springFnode[i][p][k][2] - dotSpringTangentN*tangentnode[i][p][k][2]);
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   send/recv NEB atoms to/from adjacent replicas
   received atoms matching my local atoms are stored in xprev,xnext
   replicas 0 and N-1 send but do not receive any atoms
------------------------------------------------------------------------- */


void FixNEBCAC::inter_replica_comm()
{
  int i,m;
  MPI_Request request;
  MPI_Request requestn;
  MPI_Request requests[2];
  MPI_Status statuses[2];

  // reallocate memory if necessary

  if (atom->nmax > maxlocal) reallocate();

  double **x = atom->x;
  double **f = atom->f;
  double ****xnode = atom->nodal_positions;
  double ****fnode = atom->nodal_forces;

  int *element_type = atom->element_type;
  int *poly_count = atom->poly_count;
  int **node_types = atom->node_types;
  int *nodes_count_list = atom->nodes_per_element_list; 

  int nodes_per_element;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nlocalnode = atom->maxpoly*atom->nodes_per_element * atom->nlocal;


  // -----------------------------------------------------
  // 3 cases: two for single proc per replica
  //          one for multiple procs per replica
  // -----------------------------------------------------

  // single proc per replica
  // all atoms are NEB atoms and no atom sorting
  // direct comm of x -> xprev and x -> xnext

  if (cmode == SINGLE_PROC_DIRECT) {
    //   Debug block
    // volatile int qq = 0;
    // printf("set var qq = 1");
    // while (qq == 0){}
    // if (ireplica > 0)
    //   MPI_Irecv(xprev[0],3*nlocal,MPI_DOUBLE,procprev,0,uworld,&request);
    // if (ireplica < nreplica-1) 
    //   MPI_Send(x[0],3*nlocal,MPI_DOUBLE,procnext,0,uworld);    
    // if (ireplica > 0) MPI_Wait(&request,MPI_STATUS_IGNORE);
    // if (ireplica < nreplica-1)
    //   MPI_Irecv(xnext[0],3*nlocal,MPI_DOUBLE,procnext,0,uworld,&request);
    // if (ireplica > 0)
    //   MPI_Send(x[0],3*nlocal,MPI_DOUBLE,procprev,0,uworld);
    // if (ireplica < nreplica-1) MPI_Wait(&request,MPI_STATUS_IGNORE);

    // if (ireplica < nreplica-1)
    //   MPI_Irecv(fnext[0],3*nlocal,MPI_DOUBLE,procnext,0,uworld,&request);
    // if (ireplica > 0)
    //   MPI_Send(f[0],3*nlocal,MPI_DOUBLE,procprev,0,uworld);
 
    // if (ireplica < nreplica-1) MPI_Wait(&request,MPI_STATUS_IGNORE);


    // node info send/recv
    if (ireplica > 0)
      MPI_Irecv(xprevnode[0][0][0], 3*nlocalnode, MPI_DOUBLE, procprev, 1, uworld, &requestn);
    if (ireplica < nreplica-1)
      MPI_Send(xnode[0][0][0], 3*nlocalnode, MPI_DOUBLE, procnext, 1, uworld);
    if (ireplica > 0) MPI_Wait(&requestn,MPI_STATUS_IGNORE);
    if (ireplica < nreplica-1)
      MPI_Irecv(xnextnode[0][0][0],3*nlocalnode,MPI_DOUBLE,procnext,1,uworld,&requestn);
    if (ireplica > 0)
      MPI_Send(xnode[0][0][0],3*nlocalnode,MPI_DOUBLE,procprev,1,uworld);
    if (ireplica < nreplica-1) MPI_Wait(&requestn,MPI_STATUS_IGNORE);

    if (ireplica < nreplica-1)
      MPI_Irecv(fnextnode[0][0][0],3*nlocalnode,MPI_DOUBLE,procnext,1,uworld,&requestn);
    if (ireplica > 0)
      MPI_Send(fnode[0][0][0],3*nlocalnode,MPI_DOUBLE,procprev,1,uworld);
    if (ireplica < nreplica-1) MPI_Wait(&requestn,MPI_STATUS_IGNORE);

    return;
  }


  // single proc per replica
  // but only some atoms are NEB atoms or atom sorting is enabled
  // send atom IDs and coords of only NEB atoms to prev/next proc
  // recv procs use atom->map() to match received coords to owned atoms

  if (cmode == SINGLE_PROC_MAP) {
    m = 0;
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        tagsend[m] = tag[i];
        xsend[m][0] = x[i][0];
        xsend[m][1] = x[i][1];
        xsend[m][2] = x[i][2];
        fsend[m][0] = f[i][0];
        fsend[m][1] = f[i][1];
        fsend[m][2] = f[i][2];
        m++;
      }

    if (ireplica > 0) {
      MPI_Irecv(xrecv[0],3*nebatoms,MPI_DOUBLE,procprev,0,uworld,&requests[0]);
      MPI_Irecv(tagrecv,nebatoms,MPI_LMP_TAGINT,procprev,0,uworld,&requests[1]);
    }
    if (ireplica < nreplica-1) {
      MPI_Send(xsend[0],3*nebatoms,MPI_DOUBLE,procnext,0,uworld);
      MPI_Send(tagsend,nebatoms,MPI_LMP_TAGINT,procnext,0,uworld);
    }

    if (ireplica > 0) {
      MPI_Waitall(2,requests,statuses);
      for (i = 0; i < nebatoms; i++) {
        m = atom->map(tagrecv[i]);
        xprev[m][0] = xrecv[i][0];
        xprev[m][1] = xrecv[i][1];
        xprev[m][2] = xrecv[i][2];
      }
    }
    if (ireplica < nreplica-1) {
      MPI_Irecv(xrecv[0],3*nebatoms,MPI_DOUBLE,procnext,0,uworld,&requests[0]);
      MPI_Irecv(frecv[0],3*nebatoms,MPI_DOUBLE,procnext,0,uworld,&requests[0]);
      MPI_Irecv(tagrecv,nebatoms,MPI_LMP_TAGINT,procnext,0,uworld,&requests[1]);
    }
    if (ireplica > 0) {
      MPI_Send(xsend[0],3*nebatoms,MPI_DOUBLE,procprev,0,uworld);
      MPI_Send(fsend[0],3*nebatoms,MPI_DOUBLE,procprev,0,uworld);
      MPI_Send(tagsend,nebatoms,MPI_LMP_TAGINT,procprev,0,uworld);
    }

    if (ireplica < nreplica-1) {
      MPI_Waitall(2,requests,statuses);
      for (i = 0; i < nebatoms; i++) {
        m = atom->map(tagrecv[i]);
        xnext[m][0] = xrecv[i][0];
        xnext[m][1] = xrecv[i][1];
        xnext[m][2] = xrecv[i][2];
        fnext[m][0] = frecv[i][0];
        fnext[m][1] = frecv[i][1];
        fnext[m][2] = frecv[i][2];
      }
    }

    return;
  }

  // multiple procs per replica
  // MPI_Gather all coords and atom IDs to root proc of each replica
  // send to root of adjacent replicas
  // bcast within each replica
  // each proc extracts info for atoms it owns via atom->map()

  m = 0;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      tagsend[m] = tag[i];
      xsend[m][0] = x[i][0];
      xsend[m][1] = x[i][1];
      xsend[m][2] = x[i][2];
      fsend[m][0] = f[i][0];
      fsend[m][1] = f[i][1];
      fsend[m][2] = f[i][2];
      m++;
    }

  MPI_Gather(&m,1,MPI_INT,counts,1,MPI_INT,0,world);
  displacements[0] = 0;
  for (i = 0; i < nprocs-1; i++)
    displacements[i+1] = displacements[i] + counts[i];
  MPI_Gatherv(tagsend,m,MPI_LMP_TAGINT,
              tagsendall,counts,displacements,MPI_LMP_TAGINT,0,world);
  for (i = 0; i < nprocs; i++) counts[i] *= 3;
  for (i = 0; i < nprocs-1; i++)
    displacements[i+1] = displacements[i] + counts[i];
  if (xsend) {
    MPI_Gatherv(xsend[0],3*m,MPI_DOUBLE,
                xsendall[0],counts,displacements,MPI_DOUBLE,0,world);
    MPI_Gatherv(fsend[0],3*m,MPI_DOUBLE,
                fsendall[0],counts,displacements,MPI_DOUBLE,0,world);
  } else {
    MPI_Gatherv(NULL,3*m,MPI_DOUBLE,
                xsendall[0],counts,displacements,MPI_DOUBLE,0,world);
    MPI_Gatherv(NULL,3*m,MPI_DOUBLE,
                fsendall[0],counts,displacements,MPI_DOUBLE,0,world);
  }

  if (ireplica > 0 && me == 0) {
    MPI_Irecv(xrecvall[0],3*nebatoms,MPI_DOUBLE,procprev,0,uworld,&requests[0]);
    MPI_Irecv(tagrecvall,nebatoms,MPI_LMP_TAGINT,procprev,0,uworld,
              &requests[1]);
  }
  if (ireplica < nreplica-1 && me == 0) {
    MPI_Send(xsendall[0],3*nebatoms,MPI_DOUBLE,procnext,0,uworld);
    MPI_Send(tagsendall,nebatoms,MPI_LMP_TAGINT,procnext,0,uworld);
  }

  if (ireplica > 0) {
    if (me == 0) MPI_Waitall(2,requests,statuses);

    MPI_Bcast(tagrecvall,nebatoms,MPI_INT,0,world);
    MPI_Bcast(xrecvall[0],3*nebatoms,MPI_DOUBLE,0,world);

    for (i = 0; i < nebatoms; i++) {
      m = atom->map(tagrecvall[i]);
      if (m < 0 || m >= nlocal) continue;
      xprev[m][0] = xrecvall[i][0];
      xprev[m][1] = xrecvall[i][1];
      xprev[m][2] = xrecvall[i][2];
    }
  }

  if (ireplica < nreplica-1 && me == 0) {
    MPI_Irecv(xrecvall[0],3*nebatoms,MPI_DOUBLE,procnext,0,uworld,&requests[0]);
    MPI_Irecv(frecvall[0],3*nebatoms,MPI_DOUBLE,procnext,0,uworld,&requests[0]);
    MPI_Irecv(tagrecvall,nebatoms,MPI_LMP_TAGINT,procnext,0,uworld,
              &requests[1]);
  }
  if (ireplica > 0 && me == 0) {
    MPI_Send(xsendall[0],3*nebatoms,MPI_DOUBLE,procprev,0,uworld);
    MPI_Send(fsendall[0],3*nebatoms,MPI_DOUBLE,procprev,0,uworld);
    MPI_Send(tagsendall,nebatoms,MPI_LMP_TAGINT,procprev,0,uworld);
  }

  if (ireplica < nreplica-1) {
    if (me == 0) MPI_Waitall(2,requests,statuses);

    MPI_Bcast(tagrecvall,nebatoms,MPI_INT,0,world);
    MPI_Bcast(xrecvall[0],3*nebatoms,MPI_DOUBLE,0,world);
    MPI_Bcast(frecvall[0],3*nebatoms,MPI_DOUBLE,0,world);

    for (i = 0; i < nebatoms; i++) {
      m = atom->map(tagrecvall[i]);
      if (m < 0 || m >= nlocal) continue;
      xnext[m][0] = xrecvall[i][0];
      xnext[m][1] = xrecvall[i][1];
      xnext[m][2] = xrecvall[i][2];
      fnext[m][0] = frecvall[i][0];
      fnext[m][1] = frecvall[i][1];
      fnext[m][2] = frecvall[i][2];
    }
  }
}

/* ----------------------------------------------------------------------
   reallocate xprev,xnext,tangent arrays if necessary
   reallocate communication arrays if necessary
------------------------------------------------------------------------- */

void FixNEBCAC::reallocate()
{
  maxlocal = atom->nmax;

  memory->destroy(xprev);
  memory->destroy(xnext);
  memory->destroy(tangent);
  memory->destroy(fnext);
  memory->destroy(springF);

  memory->create(xprev,maxlocal,3,"neb_CAC:xprev");
  memory->create(xnext,maxlocal,3,"neb_CAC:xnext");
  memory->create(tangent,maxlocal,3,"neb_CAC:tangent");
  memory->create(fnext,maxlocal,3,"neb_CAC:fnext");
  memory->create(springF,maxlocal,3,"neb_CAC:springF");

  //Allocate extra arrays for CAC data structures
  memory->destroy(xprevnode);
  memory->destroy(xnextnode);
  memory->destroy(tangentnode);
  memory->destroy(fnextnode);
  memory->destroy(springFnode);

  memory->create(xprevnode, maxlocal, atom->nodes_per_element, atom->maxpoly,3, "neb_CAC:xprevnode");
  memory->create(xnextnode, maxlocal, atom->nodes_per_element, atom->maxpoly,3, "neb_CAC:xnextnode");
  memory->create(fnextnode, maxlocal, atom->nodes_per_element, atom->maxpoly,3, "neb_CAC:fnextnode");
  memory->create(tangentnode, maxlocal, atom->nodes_per_element, atom->maxpoly,3, "neb_CAC:tangentnode");
  memory->create(springFnode, maxlocal, atom->nodes_per_element, atom->maxpoly,3, "neb_CAC:springFnode");


  if (cmode != SINGLE_PROC_DIRECT) {
    memory->destroy(xsend);
    memory->destroy(fsend);
    memory->destroy(xrecv);
    memory->destroy(frecv);
    memory->destroy(tagsend);
    memory->destroy(tagrecv);
    memory->create(xsend,maxlocal,3,"neb:xsend");
    memory->create(fsend,maxlocal,3,"neb:fsend");
    memory->create(xrecv,maxlocal,3,"neb:xrecv");
    memory->create(frecv,maxlocal,3,"neb:frecv");
    memory->create(tagsend,maxlocal,"neb:tagsend");
    memory->create(tagrecv,maxlocal,"neb:tagrecv");
    // TODO: add more arrays for non-single proc mode
  }

  if (NEBLongRange) {
    memory->destroy(nlenall);
    memory->create(nlenall,nreplica,"neb:nlenall");
  }
}
