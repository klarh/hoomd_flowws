
FLOWWS_CMD?=flowws_run

all: lj_spheres.sqlite dem_tetrahedra.sqlite hpmc_tetrahedra.sqlite

lj_spheres.sqlite:
	${FLOWWS_CMD} -d dump_filename $@ -d hoomd_context_args " --mode=cpu" Init --number 512 Interaction -t pair.lj -g r_cut 3 -p _ _ epsilon 1 -p _ _ sigma 1 Run --steps 1e4 --dump-period 1e3 --integrator langevin --dump-quantities 1e3 pressure,potential_energy

dem_tetrahedra.sqlite:
	${FLOWWS_CMD} -d dump_filename $@ ShapeDefinition -a shape tetrahedron -a scale 1.5 -a round 0.5 Init --number 256 DEMInteraction --type wca Run --steps 1e4 --integrator langevin --compress-to .55 Run --steps 1e5 --dump-period 1e3 --integrator nvt

hpmc_tetrahedra.sqlite:
	${FLOWWS_CMD} -d dump_filename $@ -d hoomd_context_args " --mode=cpu" ShapeDefinition -a shape tetrahedron -a scale 1.5 -a round 0.5 Init --number 256 RunHPMC --steps 1e3 --integrator nvt --compress-to .6 RunHPMC --steps 1e4 --tune .2 10 100 --integrator nvt --dump-period 1e3

.PHONY: all
