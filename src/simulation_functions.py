import openmm as mm
from openmm.app import*
from openmm import unit
import glob
import os


def setup_system(
                    base, 
                    ligand_name = None, 
                    nbmethod = 'PME', 
                    nbcutoff = 1.0,  
                    

                ):
    
    psf = CharmmPsfFile(os.path.join(base, 'step5_assembly.psf'))
    crd = CharmmCrdFile(os.path.join(base, 'step5_assembly.crd'))

    param_files = (
        glob.glob(os.path.join(param_dir, '*.rtf')) +
        glob.glob(os.path.join(param_dir, '*.prm')) +
        glob.glob(os.path.join(param_dir, '*.str'))
    )
    
    if ligand_name != None:    
        ligand_param_dir = os.path.join(base, ligand_name)
        ligand_param_files = (
            glob.glob(os.path.join(ligand_param_dir, '*.rtf')) + 
            glob.glob(os.path.join(ligand_param_dir, '*.prm'))
        )
        param_files = param_files + ligand_param_files

    params = CharmmParameterSet(param_files)

    #set the box
    sysinfo_file = os.path.join(base, "openmm", "sysinfo.dat")
    with open(sysinfo_file) as f:
        line = f.readline().strip()
        lx, ly, lz = map(float, line.strip())

    psf.setBox(lx*unit.angstroms, ly*unit.angstroms, lz*unit.angstrom)

    #create_system
    system = psf.createSystem(params, nonbondedMethod=app.LJPME, nonbondedCutoff=1.0 * unit.nanometer, constraints = app.HBonds)

    # Centering the solute within the periodic box before running the simulation
    # This step is not strictly required for the simulation to run correctly,
    # but without it, the periodic box may appear misaligned with the structure,
    # causing the protein (or membrane) to drift outside the visible box in trajectory files.
    # Centering improves visualization and helps ensure proper PBC wrapping in trajectory output.
    positions_check = crd.positions
    center = np.mean(positions_check.value_in_unit(unit.nanometer), axis=0)
    box = psf.topology.getUnitCellDimensions().value_in_unit(unit.nanometer)
    box_center = np.array(box) / 2.0
    translation = box_center - center
    centered_positions = positions_check + translation * unit.nanometer
    centered_positions = centered_positions.value_in_unit(unit.nanometer)

    return system, centered_positions



def add_hb_restraint( 
                            flag = 'protein'
                            centered_positions,
                            k = 5,
                            dim = None
                         ):
    
    if flag == 'membrane':
        constant_name = 'k_memb'
        memb = input('Please type the of the lipid (in capital): ')
    else:
        constant_name = 'k_prot'
    if dim == None:
        restraint = mm.CustomExternalForce(f'{constant_name}*periodicdistance(x, y, z, x0, y0, z0)^2')
        restraint.addGlobalParameter(constant_name, k*unit.kilocalories_per_mole/unit.angstrom**2)
        restraint.addPerParticleParameter('x0')
        restraint.addPerParticleParameter('y0')
        restraint.addPerParticleParameter('z0')

    elif dim == 'x':
        restraint = mm.CustomExternalForce(f'{constant_name}*(x - x0)^2')
        restraint.addGlobalParameter('k_memb', k*unit.kilocalories_per_mole/unit.angstrom**2)
        restraint.addPerParticleParameter('x0')

    elif dim == 'y':
        restraint = mm.CustomExternalForce(f'{constant_name}*(y - y0)^2')
        restraint.addGlobalParameter('k_memb', k*unit.kilocalories_per_mole/unit.angstrom**2)
        restraint.addPerParticleParameter('y0')

    elif dim == 'z':
        restraint = mm.CustomExternalForce(f'{constant_name}*(z - z0)^2')
        restraint.addGlobalParameter('k_memb', k*unit.kilocalories_per_mole/unit.angstrom**2)
        restraint.addPerParticleParameter('z0')
        
    system.addForce(restraint)

    if flag == 'membrane':
        for residue in psf.topology.residues():
        if residue.name == memb:
            for atom in residue.atoms():
                if atom.element.symbol != 'H':
                    pos = centered_positions[atom.index] #change made here from centered positions
                    restraint.addParticle(atom.index, [pos[0], pos[1], pos[2]])
                    print(f'{memb} atoms are now restrained.')
    else:
        std_amino_acids = ['GLY', 'TYR', 'PHE', 'ARG', 'HIS', 'ALA', 'PRO', 'GLU', 'SER', 'LYS',
                           'THR', 'MET', 'CYS', 'LEU', 'GLN', 'ASN', 'VAL', 'ILE', 'ASP', 'TRP']
        for residue in psf.topology.residues():
            if residue.name in std_amino_acids:
                for atom in residue.atoms():
                    if atom.element.symbol != 'H':
                        pos = centered_positions[atom.index]
                        protein_restraint.addParticle(atom.index, [pos[0], pos[1], pos[2]])
                        print('Protein atoms are now restrained.')
        

def define_platform(                    
                    platform_name = 'CUDA'
                    num_of_gpus = 3,
                    ):
    platform = mm.Platform.getPlatformByName('CUDA')
    #to run the simulation on several GPU's parallelly
    properties  = {'CudaDeviceIndex': ",".join(str(i) for i in range(num_of_gpus)) }

    return platform, properties


def get_latest_file(
                            folder_path,
                            flag = 'state'
                         ):
    # Regex to match files like x0_123.xml
    if flag == 'state':
        pattern = re.compile(r"x0_(\d+)\.xml$")
    elif flag == 'checkpoint':
        pattern = re.compile(r"check_(\d+)\.chk$")
    elif flag == 'trajectory':
        pattern = re.compile(r"traj_(\d+)\.chk$")

    latest_file = None
    latest_num = -1

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > latest_num:
                latest_num = num
                latest_file = filename

    if latest_file:
        return latest_num, os.path.join(folder_path, latest_file)
    else:
        return None


def simulate(
                system, positions = setup_system(),
                T = 310,
                gamma = 1.0,
                dt = 0.002,
                nsteps = 500000,
                n_frames = 1000,
                integrator = 'Langevin',
                platform, properties = define_platform(),
                states_dir = 'states/',
                traj_dir = 'traj/',
                system_dir = 'system/',
                continue_from_prev = None,
                traj_file = 'traj_1.dcd',
                chk_file = 'check_1.chk',
                ensemble = 'NVT'
                
            ):

    if integrator == 'Langevin':
        integrator_eq = mm.LangevinIntegrator(T*kelvin, gamma/picoseconds, dt*picoseconds)
    elif integrator == 'Langevinmiddle':
        integrator_eq = mm.LangevinMiddleIntegrator(T*kelvin, gamma/picoseconds, dt*femtoseconds)
    
    simulation_eq = app.Simulation(psf.topology, system, integrator_eq, platform, properties)

    latest_num, prev_state_file = get_latest_file(states_dir, 'state')
    traj_num, traj_file = get_latest_file(traj_dir, 'trajectory')
    chk_num, chk_file = get_latest_file(traj_dir, 'checkpoint')

    
    if continue_from_prev == 'state':
        print('Loading the ')
        simulation_eq.loadState(prev_state_file)
    elif continue_from_prev == 'checkpoint':
        simulation_eq.loadCheckpoint(chk_file)
    else:
        simulation_eq.context.setPositions(positions)
        simulation_eq.context.setVelocitiesToTemperature(T*kelvin)


        if ensemble == 'NPT':
            P = float(input("Please enter the pressure value (in atm) for the ensemble: \n"))
            for i in reversed(range(system.getNumForces()):
                force = system.getForce(i)
                if isinstance(force, mm.MonteCarloBarostat) or isinstance(force, mm.MonteCarloAnisotropicBarostat):
                    system.removeForce(i)
            system.addForce(mm.MonteCarloBarostat(P*unit.bar, T*unit.kelvin))
        elif ensemble == 'NPgT':
            for i in reversed(range(system.getNumForces()):
                force = system.getForce(i)
                if isinstance(force, mm.MonteCarloBarostat) or isinstance(force, mm.MonteCarloAnisotropicBarostat):
                    system.removeForce(i)
            






    

    state_1 = simulation_eq.context.getState(getForces=True)
    forces = state_1.getForces(asNumpy = True).value_in_unit(unit.kilojoule/unit.nanometer/unit.mole)
    norm = np.linalg.norm(forces, axis = 1)
    print(f'Maximum force in the system before this part of simulation (kilojoule/nanometer/mole): {np.max(norm)}')


    simulation_eq.reporters.append(app.DCDReporter(traj_dir + 'traj_' + str(traj_num + 1), nsteps//n_frames, enforcePeriodicBox=True))
    
    simulation_eq.reporters.append(app.CheckpointReporter(traj_dir + 'check_' + str(chk_num + 1), nsteps//n_frames))

    simulation_eq.step(nsteps)

    state_2 = simulation_eq.context.getState(getForces=True)
    forces = state_2.getForces(asNumpy = True).value_in_unit(unit.kilojoule/unit.nanometer/unit.mole)
    norm = np.linalg.norm(forces, axis = 1)
    print(f'\nMaximum force in the system after this part of simulation (kilojoule/nanometer/mole): {np.max(norm)}')
    
    
    





    