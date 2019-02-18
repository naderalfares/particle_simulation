#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include<vector>


#define density 0.0005
#define NUM_CELLS 4




void apply_forces_to_cell(std::vector<particle_t*> src, std::vector<particle_t*> & cell, int* navg, double* dmin, double* davg){
    for(int i = 0; i < src.size(); i++)
        for(int j = 0; j < cell.size(); j++){
            src[i]->ax = src[i]->ay = 0;
		    apply_force(*(src[i]), *(cell[j]),dmin,davg,navg);
        }

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
        

    double size = density*n;
    double cell_size = size/NUM_CELLS;
    //assume that grid is NUM_CELLS x NUM_CELLS
    //
    //XXX: use malloc instead of iteration
    std::vector<particle_t*> cells[NUM_CELLS][NUM_CELLS]; 
    for(int i = 0; i < NUM_CELLS; i++){
        for(int j=0; j < NUM_CELLS; j++){
            cells[i][j] = std::vector<particle_t* >();
        }
    }

    for(int i = 0; i < n; i++){
        //TODO: partition particles into cells
        int cell_i = floor(particles[i].x / cell_size);
        int cell_j = floor(particles[i].y / cell_size);
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

        // find nearby cells
        for(int i = 0; i < NUM_CELLS; i++){
            for(int j = 0; j < NUM_CELLS; j++){


                std::vector<std::vector<particle_t* > > neighbor_cells;
                //collect neighbors
                for (int ii = i - 1; ii <= i + 1; ii++) 
                    for (int jj = j - 1; jj <= j + 1; jj++) {
                        //check if it is a possible cell
                        if (i >= 0 && i < NUM_CELLS && j >= 0 && j < NUM_CELLS)
                            neighbor_cells.push_back(cells[i][j]);
                    }


                //compute forces of close cells to the current cell
                for (int p_index = 0; p_index < cells[i][j].size(); p_index++){
                    cells[i][j][p_index]->ax = 0;
                    cells[i][j][p_index]->ay = 0;

                    for (int n_index = 0; n_index < neighbor_cells.size(); n_index++ ){
                        for (int np_index = 0; np_index < neighbor_cells[n_index].size(); np_index++){
                          apply_force(*cells[i][j][p_index], *neighbor_cells[n_index][np_index],&dmin, &davg, &navg);
                        }
                    }
                }   
            }
        }


        //
        //  move particles
        //
        for( int i = 0; i < n; i++ ) 
            move( particles[i] );		


        // clearing the bins
        
        for(int i = 0; i < NUM_CELLS; i++){
            for(int j=0; j < NUM_CELLS; j++){
                cells[i][j].clear();
            }
        }


        // re-constructing bins
        for(int i = 0; i < n; i++){
            //TODO: partition particles into cells
            int cell_i = floor(particles[i].x / cell_size);
            int cell_j = floor(particles[i].y / cell_size);

            cells[cell_i][cell_j].push_back(&particles[i]); 
        }

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
    }
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



