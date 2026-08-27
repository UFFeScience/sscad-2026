nextflow.enable.dsl=2

/*
 * Uso:
 *   nextflow run main.nf --app kmeans
 *   nextflow run main.nf --app matmul
 *   nextflow run main.nf --app lysozyme
 *
 * Requisitos:
 *   - KMeans/Matmul: python3 + numpy
 *   - Lysozyme: GROMACS com comandos gmx e gmx_mpi disponíveis
 */

params.app = 'kmeans'
params.outdir = 'results'

// KMeans
params.kmeans_n = 10000
params.kmeans_dim = 8
params.kmeans_k = 4
params.kmeans_iter = 100
params.kmeans_tol = 1e-4
params.kmeans_seed = 42

// Matmul
params.matmul_n = 1024
params.matmul_m = 1024
params.matmul_p = 1024
params.matmul_seed = 42

// Lysozyme / GROMACS
params.pdb_id = '1AKI'
params.gmx_cmd = 'gmx'
params.gmx_mpi_cmd = 'mpirun -np 2 gmx_mpi'

params.forcefield = 'amber99sb-ildn'
params.water = 'tip3p'
params.box_distance = 1.0

params.em_steps = 5000
params.md_steps = 5000
params.temperature = 300
params.omp_threads = 1


process FETCH_PDB {

    publishDir "${params.outdir}/lysozyme/input", mode: 'copy', overwrite: true

    output:
    path "lysozyme.pdb"

    script:
    """
    curl -L --fail -o lysozyme.pdb https://files.rcsb.org/download/${params.pdb_id}.pdb \\
      || wget -O lysozyme.pdb https://files.rcsb.org/download/${params.pdb_id}.pdb
    """
}


process LYSOZYME_MD {

    publishDir "${params.outdir}/lysozyme", mode: 'copy', overwrite: true

    input:
    path pdb

    output:
    path "lysozyme_outputs"

    script:
    """
    set -euo pipefail

    export OMP_NUM_THREADS=${params.omp_threads}

    cp ${pdb} input.pdb

    cat > ions.mdp <<'EOF'
integrator              = steep
emtol                   = 1000.0
emstep                  = 0.01
nsteps                  = 500
cutoff-scheme           = Verlet
coulombtype             = PME
rcoulomb                = 1.0
rvdw                    = 1.0
pbc                     = xyz
EOF

    cat > minim.mdp <<EOF
integrator              = steep
emtol                   = 1000.0
emstep                  = 0.01
nsteps                  = ${params.em_steps}
cutoff-scheme           = Verlet
coulombtype             = PME
rcoulomb                = 1.0
rvdw                    = 1.0
pbc                     = xyz
EOF

    cat > nvt.mdp <<EOF
define                  = -DPOSRES
integrator              = md
nsteps                  = ${params.md_steps}
dt                      = 0.002

nstxout                 = 0
nstvout                 = 0
nstenergy               = 500
nstlog                  = 500
nstxout-compressed      = 500

continuation            = no
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4

cutoff-scheme           = Verlet
nstlist                 = 20
rlist                   = 1.0
coulombtype             = PME
rcoulomb                = 1.0
vdwtype                 = Cut-off
rvdw                    = 1.0

tcoupl                  = V-rescale
tc-grps                 = Protein Non-Protein
tau_t                   = 0.1 0.1
ref_t                   = ${params.temperature} ${params.temperature}

pcoupl                  = no
pbc                     = xyz

DispCorr                = EnerPres
gen_vel                 = yes
gen_temp                = ${params.temperature}
gen_seed                = -1
EOF

    echo "=== 1. Gerando topologia com pdb2gmx ==="
    ${params.gmx_cmd} pdb2gmx \\
        -f input.pdb \\
        -o processed.gro \\
        -p topol.top \\
        -water ${params.water} \\
        -ff ${params.forcefield} \\
        -ignh

    echo "=== 2. Definindo caixa ==="
    ${params.gmx_cmd} editconf \\
        -f processed.gro \\
        -o boxed.gro \\
        -c \\
        -d ${params.box_distance} \\
        -bt cubic

    echo "=== 3. Solvatando ==="
    ${params.gmx_cmd} solvate \\
        -cp boxed.gro \\
        -cs spc216.gro \\
        -o solvated.gro \\
        -p topol.top

    echo "=== 4. Preparando adição de íons ==="
    ${params.gmx_cmd} grompp \\
        -f ions.mdp \\
        -c solvated.gro \\
        -p topol.top \\
        -o ions.tpr \\
        -maxwarn 2

    echo "=== 5. Adicionando íons ==="
    printf "SOL\\n" | ${params.gmx_cmd} genion \\
        -s ions.tpr \\
        -o solv_ions.gro \\
        -p topol.top \\
        -pname NA \\
        -nname CL \\
        -neutral

    echo "=== 6. Preparando minimização de energia ==="
    ${params.gmx_cmd} grompp \\
        -f minim.mdp \\
        -c solv_ions.gro \\
        -p topol.top \\
        -o em.tpr \\
        -maxwarn 2

    echo "=== 7. Rodando minimização com gmx_mpi ==="
    ${params.gmx_mpi_cmd} mdrun \\
        -deffnm em \\
        -ntomp ${params.omp_threads}

    echo "=== 8. Preparando dinâmica NVT curta ==="
    ${params.gmx_cmd} grompp \\
        -f nvt.mdp \\
        -c em.gro \\
        -r em.gro \\
        -p topol.top \\
        -o nvt.tpr \\
        -maxwarn 2

    echo "=== 9. Rodando dinâmica NVT curta com gmx_mpi ==="
    ${params.gmx_mpi_cmd} mdrun \\
        -deffnm nvt \\
        -ntomp ${params.omp_threads}

    mkdir -p lysozyme_outputs

    cp -f *.mdp lysozyme_outputs/ 2>/dev/null || true
    cp -f *.top lysozyme_outputs/ 2>/dev/null || true
    cp -f *.itp lysozyme_outputs/ 2>/dev/null || true
    cp -f *.gro lysozyme_outputs/ 2>/dev/null || true
    cp -f *.tpr lysozyme_outputs/ 2>/dev/null || true
    cp -f *.log lysozyme_outputs/ 2>/dev/null || true
    cp -f *.edr lysozyme_outputs/ 2>/dev/null || true
    cp -f *.xtc lysozyme_outputs/ 2>/dev/null || true
    cp -f *.cpt lysozyme_outputs/ 2>/dev/null || true
    """
}


workflow LYSOZYME_WORKFLOW {
    main:
    FETCH_PDB()
    LYSOZYME_MD(FETCH_PDB.out)
}


workflow {
    LYSOZYME_WORKFLOW()
}
