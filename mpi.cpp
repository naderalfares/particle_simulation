#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"
#include<vector>
#include<math.h>

// TODO: keep the cell_size fixed at 0.1 so that the entire space is integrally divisible by cell size
#define CELL_SIZE 0.01
#define density 0.0005


//XXX:
struct proc_info {
    float xLow, xHigh;
    float yLow, yHigh;
    float gxHigh, gxLow; // CELL_SIZE if at edge else 0
    float gyHigh, gyLow; // same
};


enum bin_status {GHOST, INNER};
struct bin{
    std::vector<particle_t*> particles;
    bin_status status;     //status
    float xLow, xHigh;
    float yLow, yHigh; 
};



void findMeAndMyNeighbors(const std::vector<std::vector<bin>> bins,int i_dim, int j_dim, int bin_i, int bin_j
                        ,std::vector<bin> &neighbors){
     for(int i = -1; i < 2; i++){
            int nrow = bin_i - i;
        for( int j = -1; j < 2; j++){
            int ncol = bin_j - j;
            if(nrow >= 0 && ncol >= 0 && nrow < i_dim && ncol < j_dim){
                neighbors.push_back(bins[i][j]);
            }
        }
    }    
}
        



// particles: the particles in the boundary of the processor and not the ghost particles
void packing(particle_t** particles, int n, struct proc_info * procs_info, int n_proc,
                int** partition_offset, int** partition_sizes){
    
    particle_t* new_particles = (particle_t *) malloc(n * sizeof(particle_t));
    int index = 0, count = 0;
    for(int i = 0; i < n_proc; i ++){
        *partition_offset[i] = index;
        count = 0;
        int xHi = procs_info[i].xHigh + procs_info[i].gxHigh;
        int xLo = procs_info[i].xLow + procs_info[i].gxLow;
        int yHi = procs_info[i].yHigh + procs_info[i].gyHigh;
        int yLo = procs_info[i].yLow + procs_info[i].gyLow;
        for(int j = 0; j < n; j++){
            int xCord = particles[j]->x;
            int yCord = particles[j]->y;
            // compare particle coordiantes to processor boundaries
            if(xCord < xHi and xCord > xLo and yCord < yHi and yCord > yLo ) {
                new_particles[index++] = *particles[j];
                count++;
            }
        }
       *partition_sizes[i] = count;
    }

   free(*particles);
   particles = &new_particles;
   
};

void communicateData(particle_t **particles,  int **partitionSizes, int *partitionOffsets, int nprocs, MPI_Datatype PARTICLE) {
     int numParticles = 0;
     // calculate the total particles = sum of all the particles to be send including the ones replicated
     for(int i=0; i < nprocs; i++)
        numParticles += *partitionSizes[i];

     // receive the new particles after all to all
     particle_t *newParticles = (particle_t*)malloc(sizeof(particle_t) * numParticles);
     int *newPartitionSizes = (int *)malloc(sizeof(int) * nprocs);

     MPI_Alltoallv( *particles, *partitionSizes,
		    partitionOffsets, PARTICLE, newParticles,
		    newPartitionSizes, NULL, PARTICLE,
		    MPI_COMM_WORLD
	    	  );
     free(*particles);
     free(*partitionSizes);
     particles = &newParticles;
     partitionSizes = &newPartitionSizes;
}

// The first initialized particles and proc_info will be scattered to the respective processors
void initialParticleScatter(particle_t **particles, int &numParticles, int **partitionSizes, int **partitionOffsets,
                    int numProcs,struct proc_info* pinfo, int rank, MPI_Datatype PARTICLE) {
    // only rank 0 has valid particles others wil have null
    if(*partitionSizes == NULL)
	*partitionSizes = (int *)malloc (sizeof(int) * numProcs);
    if(*partitionOffsets == NULL)
        *partitionOffsets = (int *) malloc(sizeof(int) * numProcs);

    // create the partitions for each processor to be received 
    if(rank==0)
	packing(particles, numParticles, pinfo, numProcs,
                partitionOffsets, partitionSizes);
    particle_t *local = (particle_t *)malloc(sizeof(particle_t) * numParticles);
    int nlocal = numParticles/numProcs;
    MPI_Scatterv( particles, (const int *)partitionSizes, (const int *)partitionOffsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );
    *particles = local;
    numParticles = nlocal;
}


void initializeGrid(const struct proc_info p_info, std::vector<std::vector<struct bin>> &bins, int dim_x, int dim_y){
    // used vectros instead of arrays since cpp requires knowing n-1 dimensions when passing by refrence
    float start_x, start_y;
    bin_status status;
    if(p_info.gxLow > 0){
        start_x = p_info.gxLow;
        status = GHOST;
    } else {
        start_x = p_info.xLow;
        status = INNER;
    }
    
    for(int j = 0; j < dim_x; j++){
        // this is checking for the right edge
        if(start_x > p_info.xHigh){
            status = GHOST;
        //need to re-assign for rows     
        if(p_info.gyHigh > 0){
            start_y = p_info.gyHigh;
            status = GHOST;
        }else{
            start_y = p_info.yHigh;
        }
        for(int i = 0; i < dim_y; i++){
            //check if the x limit is in the inner bins but y is not
            // if x is already in the ghost, then no need to check for y
            if(dim_y - i == 1 && p_info.gyLow > 0){
                status = GHOST;
            }
            bins[i][j].status = status;
            // incremeant in column
            start_y += CELL_SIZE;
        }
        start_x += CELL_SIZE;
        status = INNER;
        }
    }

}


/*
 * intialize proc_info on all the processors for the first time only
 * based on the following represented boundary condition
 *      yLow
 *	    ________
 *	    |      |
 * xLow	|      | xHigh
 *	    |      |
 *	    --------
 *	    yHigh
 * */

void initializeProcInfo(struct proc_info **proc_info, int space) {
   int procPointer = 0;
   for(int i=0; i < space; i+=CELL_SIZE) {
      struct proc_info *info = &((*proc_info)[procPointer]);
      info->yLow = i * CELL_SIZE;
      info->yHigh = (i+1) * CELL_SIZE;
      info->xLow = i * CELL_SIZE;
      info->xHigh = (i+1) * CELL_SIZE;
      if(info->yLow == 0) 
        info->gyLow = 0;
      else
        info->gyLow = -CELL_SIZE;
      if(info->yHigh == space) 
	info->gyHigh = 0;
      else
        info->gyHigh = CELL_SIZE;
      if(info->xLow == 0) 
        info->gxLow = 0;
      else
        info->gxLow = -CELL_SIZE;
      if(info->xHigh==space) 
	info->gxHigh=0;
      else 
        info->gxHigh = CELL_SIZE;
   }
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
 
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
   


    //XXX: the value on these don't matter, they are going to change
    //      when init_scatter function is called 
    //
    //  set up the data partitioning across processors
    //
    int particle_per_proc = (n + n_proc - 1) / n_proc;
    int *partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
    for( int i = 0; i < n_proc+1; i++ )
        partition_offsets[i] = min( i * particle_per_proc, n );
    
    int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
    for( int i = 0; i < n_proc; i++ )
        partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];
    
    //
    //  allocate storage for local partition
    //
    int nlocal = partition_sizes[rank];
    particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );
    
    //
    //  allcoate array of proc info in every processor
    //  
    
    const double size = sqrt(density * n);
    const int numCells = ceil(size/CELL_SIZE);
    proc_info*  procs_info = (proc_info*) malloc( n_proc * sizeof(proc_info));
    initializeProcInfo(&procs_info, size);


    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size( n );
    if( rank == 0 )
        init_particles( n, particles );
    //MPI_Scatterv( particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );
    //scatter particles 
    initialParticleScatter(&particles, n, &partition_sizes, &partition_offsets, n_proc, &procs_info[0], 0 , PARTICLE);
    nlocal = n;
    local  = particles;
    
    //  initilize local bins
    //
    
    struct proc_info myInfo = procs_info[rank];
    int    j_dim            = ( (myInfo.xHigh + myInfo.gxHigh) - (myInfo.xLow - myInfo.gxLow) ) / CELL_SIZE;
    int    i_dim            = ( (myInfo.yHigh + myInfo.gyHigh) - (myInfo.yLow - myInfo.gyLow) ) / CELL_SIZE;
    //  used truncating vectors for passing by refrence simplicity
    std::vector<std::vector<bin>> bins (i_dim, std::vector<bin> (j_dim, bin()));
    

    // this procs info
    float myX_high, myX_low;
    float myY_high, myY_low;
    //TODO: put in module
    if(procs_info[rank].gxHigh > 0){
        myX_high = procs_info[rank].gxHigh;
    } else {
        myX_high = procs_info[rank].xHigh;
    }
    
    if(procs_info[rank].gxLow > 0){
        myX_low = procs_info[rank].gxLow;
    } else {
        myX_low = procs_info[rank].xLow;
    }

    if(procs_info[rank].gyHigh > 0){
        myY_high = procs_info[rank].gyHigh;
    } else {
        myY_high = procs_info[rank].yHigh;
    }

    if(procs_info[rank].gyLow > 0){
        myY_low = procs_info[rank].gyLow;
    } else {
        myY_low = procs_info[rank].yLow;
    }
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    
    //XXX: delete before testing
    int temp_nsteps = 5;
    for( int step = 0; step < temp_nsteps; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;
        
        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
          if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );

        //TODO: bin particles given my bin
        for( int i = 0; i < nlocal; i ++){
            float relative_x = local[i].x - myX_low;
            float relative_y = local[i].y - myY_high;
            assert(relative_x > 0 && relative_y > 0);
            int bin_i = floor(relative_y/CELL_SIZE);
            int bin_j = floor(relative_x/CELL_SIZE);
            bins[bin_i][bin_j].particles.push_back(&local[i]);   
        }
        
        //
        //  compute all forces
        //
        std::vector<bin> neighbors;
        
        for(int j = 0; j < j_dim; j++){
            for( int i = 0; i < i_dim; i++){
                if(bins[i][j].status == INNER){
                    findMeAndMyNeighbors(bins, i_dim, j_dim, i, j, neighbors);
                    //TODO: Apply forces
                }
            }
        }
        // clear neighbor bins
        for(int i = 0; i < i_dim; i ++)
            for( int j = 0; j < j_dim; j++)
                if(bins[i][j].status == GHOST)
                    bins[i][j].particles.clear();
       

        
        if( find_option( argc, argv, "-no" ) == -1 )
        {
          
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

 
          if (rank == 0){
            //
            // Computing statistical data
            //
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
          }
        }

        //
        //  move particles
        //
        //TODO: aggragate all particles that have moved from inner bin 
        for( int i = 0; i < nlocal; i++ )
            move( local[i] );
    }
    simulation_time = read_timer( ) - simulation_time;
  
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

      if( find_option( argc, argv, "-no" ) == -1 )
      {
        if (nabsavg) absavg /= nabsavg;
      // 
      //  -the minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
      //
      //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
      //
      printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
      if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
      if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
      }
      printf("\n");     
        
      //  
      // Printing summary data
      //  
      if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    free( partition_offsets );
    free( partition_sizes );
    free( local );
    free( particles );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
