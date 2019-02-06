# Use these commands to generate the LAMMPS input script and data file
# (and other auxilliary files):


# Create LAMMPS input files this way:
cd moltemplate_files

  # run moltemplate

  moltemplate.sh system.lt

  # This will generate various files with names ending in *.in* and *.data.
  # These files are the input files directly read by LAMMPS.  Move them to
  # the parent directory (or wherever you plan to run the simulation).

  mv -f system.in* system.data ../

  # The "table_int.dat" file contains tabular data for the lipid INT-INT atom
  # 1/r^2 interaction.  We need it too. (This slows down the simulation by x2,
  # so I might look for a way to get rid of it later.)
  cp -f tabulated_potential.dat ../

  # Optional:
  # The "./output_ttree/" directory is full of temporary files generated by
  # moltemplate.  They can be useful for debugging, but are usually thrown away.
  rm -rf output_ttree/

cd ../
