#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include<vector>


#define density 0.0005
#define CELL_SIZE 0.02


// Function applies forces of particles of one cell to particles in another cell
void apply_forces_to_cell(std::vector<particle_t*> &src, std::vector<particle_t*> & cell, int* navg, double* dmin, double* davg){
    for(int i = 0; i < src.size(); i++) {
	//int xOrg = src[i]->x;
	//int yOrg = src[i]->y;
      	for(int j = 0; j < cell.size(); j++)
		    apply_force(*(src[i]), *(cell[j]),dmin,davg,navg);
	//assert(src[i]->x != xOrg);    
	//assert(src[i]->y != yOrg);    
    }
}

void initCellParticles(std::vector<particle_t*> &src) {
    for(int i = 0; i < src.size(); i++) 
        src[i]->ax = src[i]->ay = 0;
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg,nabsavg=0;
    double davg,dmin, absmin=1.0, absavg=0.0;

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
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    
    init_particles( n, particles );
        

    double size = sqrt(density * n);
    int numCells = ceil(size/CELL_SIZE);
    //assume that grid is numCells x numCells
    //
    //XXX: use malloc instead of iteration
    std::vector<particle_t*> cells[numCells][numCells]; 
    // for(int i = 0; i < numCells; i++){
    //     for(int j=0; j < numCells; j++){
    //         cells[i][j] = std::vector<particle_t* >();
    //     }
    // }

    for(int i = 0; i < n; i++){
        //TODO: partition particles into cells
        int cell_i = floor(particles[i].x / CELL_SIZE);
        int cell_j = floor(particles[i].y / CELL_SIZE);
        cells[cell_i][cell_j].push_back(&particles[i]); 
    }
    
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
	
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        davg = 0.0;
	dmin = 1.0;
        //
        //  compute forces
        //

        
        for(int i = 0; i < numCells; i++){
            for(int j = 0; j < numCells; j++){
		initCellParticles(cells[i][j]); // initialize particles in current cell
                //unrolling
                //TODO: apply forces for subgrids
                      apply_forces_to_cell(cells[i][j],cells[i][j], &navg, &dmin, &davg); 
                if(i==0 && j==0){
                      apply_forces_to_cell(cells[i][j],cells[1][0], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[0][1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[1][1], &navg, &dmin, &davg);
                } else if(i == numCells -1 && j == numCells - 1){
                      apply_forces_to_cell(cells[i][j],cells[i-1][j-1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i-1][j], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i][j-1], &navg, &dmin, &davg);
                
                } else if(i == numCells - 1 && j == 0){
                      apply_forces_to_cell(cells[i][j],cells[i][j+1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i-1][j], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i-1][j+1], &navg, &dmin, &davg);

                } else if(i == 0 && j == numCells -1){
                      apply_forces_to_cell(cells[i][j],cells[i][j-1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i+1][j-1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i+1][j], &navg, &dmin, &davg);
                }else if(i == numCells -1){
                      apply_forces_to_cell(cells[i][j],cells[i][j+1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i][j-1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i-1][j+1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i-1][j-1], &navg, &dmin, &davg);
                      apply_forces_to_cell(cells[i][j],cells[i-1][j], &navg, &dmin, &davg); 
                      
                }else if(j == numCells -1){
                      apply_forces_to_cell(cells[i][j],cells[i+1][j], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i+1][j-1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i-1][j], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i-1][j-1], &navg, &dmin, &davg);
                      apply_forces_to_cell(cells[i][j],cells[i][j-1], &navg, &dmin, &davg); 
                
                }else if(i == 0){
                      apply_forces_to_cell(cells[i][j],cells[i][j-1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i][j+1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i+1][j], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i+1][j-1], &navg, &dmin, &davg);
                      apply_forces_to_cell(cells[i][j],cells[i+1][j+1], &navg, &dmin, &davg); 
                      
                }else if(j == 0){
                      apply_forces_to_cell(cells[i][j],cells[i-1][j], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i+1][j], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i-1][j+1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i][j+1], &navg, &dmin, &davg);
                      apply_forces_to_cell(cells[i][j],cells[i+1][j+1], &navg, &dmin, &davg); 
                }else{
                      apply_forces_to_cell(cells[i][j],cells[i][j+1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i][j-1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i-1][j+1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i-1][j], &navg, &dmin, &davg);
                      apply_forces_to_cell(cells[i][j],cells[i-1][j-1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i+1][j+1], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i+1][j], &navg, &dmin, &davg); 
                      apply_forces_to_cell(cells[i][j],cells[i+1][j-1], &navg, &dmin, &davg); 
                }

            } // end of j loop
        } // end of i loop


        //
        //  move particles
        //
        for( int i = 0; i < n; i++ ) 
            move( particles[i] );		


        // clearing the bins
        
        for(int i = 0; i < numCells; i++){
            for(int j=0; j < numCells; j++){
                cells[i][j].clear();
            }
        }
	

        // re-constructing bins
        for(int i = 0; i < n; i++){
            //TODO: partition particles into cells
            int cell_i = floor(particles[i].x / CELL_SIZE);
            int cell_j = floor(particles[i].y / CELL_SIZE);

            cells[cell_i][cell_j].push_back(&particles[i]); 
        }

	// assert for no of particles
/*	int sum  = 0;
        for(int i = 0; i < numCells; i++){
            for(int j=0; j < numCells; j++){
                sum += cells[i][j].size();
            }
        }
	assert(sum==n);*/

        if( find_option( argc, argv, "-no" ) == -1 )
        {
          //
          // Computing statistical data
          //
          if (navg) {
            absavg +=  davg/navg;
            nabsavg++;
          }
          if (dmin < absmin) absmin = dmin;
		
          //
          //  save if necessary
          //
          if( fsave && (step%SAVEFREQ) == 0 )
              save( fsave, n, particles );
        }
    } // end of NSTEPS iteration

    simulation_time = read_timer( ) - simulation_time;
    
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
        fprintf(fsum,"%d %g\n",n,simulation_time);
 
    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );    
    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}



