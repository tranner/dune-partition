#include <config.h>

// iostream includes
#include <iostream>

// include grid part
#include <dune/fem/gridpart/adaptiveleafgridpart.hh>

// include output
#include <dune/fem/io/file/dataoutput.hh>

// include grid width
#include <dune/fem/misc/gridwidth.hh>

// include norms
#include <dune/fem/misc/l2norm.hh>

// include header of adaptive scheme
#include "partition.hh"

#include "heatmodel.hh"
#include "partitionscheme.hh"

// assemble-solve-estimate-mark-refine-IO-error-doitagain
template <class HGridType>
double algorithm ( HGridType &grid, int step )
{
  typedef Dune::FunctionSpace< double, double, HGridType :: dimensionworld, 1 > FunctionSpaceType;

  // create time provider
  Dune::Fem::GridTimeProvider< HGridType > timeProvider( grid );

  // we want to solve the problem on the leaf elements of the grid
  typedef Dune::Fem::AdaptiveLeafGridPart< HGridType, Dune::InteriorBorder_Partition > GridPartType;
  GridPartType gridPart(grid);

  // type of the mathematical model used
  typedef PartitionProblem< FunctionSpaceType > ProblemType;
  typedef HeatModel< FunctionSpaceType, GridPartType > ModelType;

  ProblemType problem( timeProvider ) ;

  // implicit model for left hand side
  ModelType implicitModel( problem, gridPart, true );

  // explicit model for right hand side
  ModelType explicitModel( problem, gridPart, false );

  // create partition scheme
  typedef PartitionScheme< ModelType, ModelType > SchemeType;
  SchemeType scheme( gridPart, implicitModel, explicitModel );

  typedef Dune::Fem::GridFunctionAdapter< ProblemType, GridPartType > GridExactSolutionType;
  GridExactSolutionType gridExactSolution("exact solution", problem, gridPart, 5 );
  //! input/output tuple and setup datawritter
  typedef Dune::tuple< const typename SchemeType::DiscreteFunctionType *, GridExactSolutionType * > IOTupleType;
  typedef Dune::Fem::DataOutput< HGridType, IOTupleType > DataOutputType;
  IOTupleType ioTuple( &(scheme.solution()), &gridExactSolution) ; // tuple with pointers
  DataOutputType dataOutput( grid, ioTuple, timeProvider, DataOutputParameters( step ) );

  const double endTime  = Dune::Parameter::getValue< double >( "partition.endtime", 2.0 );
  const double dtreducefactor = Dune::Parameter::getValue< double >("partition.reducetimestepfactor", 1 );
  double timeStep = Dune::Parameter::getValue< double >( "partition.timestep", 0.125 );

  timeStep *= pow(dtreducefactor,step);

  // initialize with fixed time step
  timeProvider.init( timeStep ) ;

  // initialize scheme and output initial data
  scheme.initialize();
  scheme.normalise();
  // write initial solve
  dataOutput.write( timeProvider );

  //! type of restriction/prolongation projection for adaptive simulations
  //! (use default here, i.e. LagrangeInterpolation)
  typedef Dune::Fem::RestrictProlongDefault< typename SchemeType :: DiscreteFunctionType >  RestrictionProlongationType;

  //! type of adaptation manager handling adapation and DoF compression
  typedef Dune::Fem::AdaptationManager< HGridType, RestrictionProlongationType > AdaptationManagerType;

  RestrictionProlongationType restrictProlong( scheme.solution() );
  AdaptationManagerType adaptationManager( grid, restrictProlong );
  bool adapter = true;

  // reduction parameters
  const double epsFact = Dune::Parameter::getValue< double >( "partition.epsilonfactor", std::sqrt(0.5) );
  const double tauFact = Dune::Parameter::getValue< double >( "partition.timestepfactor", 1 );
  
  // set tolerances
  const double innerTol = Dune::Parameter::getValue< double >( "partition.innertol", 1.0e-4 );
  const double outerTol = Dune::Parameter::getValue< double >( "partition.outertol", 1.0e-8 );

  const double nextEnergyUpdate = Dune::Parameter::getValue< double >( "partition.nextenergyupdate", 0.1 );

    // set helper variables
  double energyOld = -1.0;
  double energyPrerefine = -1.0;
  double nextEnergyTime = nextEnergyUpdate;

  // time loop, increment with fixed time step
  for( ; timeProvider.time() < endTime ; timeProvider.next( timeStep ) )
  {
    // assemble explicit pare
    scheme.prepare();
    // solve once (we need to assemble the system the first time the method
    // is called)
    scheme.solve( adapter );
    adapter = false;
    scheme.solveODE();
    scheme.normalise();
    dataOutput.write( timeProvider );

    if( timeProvider.time() > nextEnergyTime )
      {
	// compute energy and Seps
	double Seps;
	const double energy = scheme.energy( Seps );
	// output on root process
	if(  Dune::MPIManager::rank() == 0 )
	  std::cout << "  time: " << timeProvider.time() << "\t"
		    << "energy: " << energy << "\t"
		    << "Seps: " << Seps << std::endl;

	// test for convergence
	if( std::abs( energy - energyOld ) < innerTol )
	  {
	    // print diagnostics
	    const double h = Dune :: GridWidth :: calcGridWidth( gridPart );
	    if(  Dune::MPIManager::rank() == 0 )
	      std::cout << "h: " << h << "\t"
			<< "eps: " << scheme.epsilon() << "\t"
			<< "energy: " << energy << "\t"
			<< "Seps: " << Seps;

	    // test for errors in known cases
	    if( Dune :: MPIManager :: size() == 2 || Dune :: MPIManager :: size() == 3 )
	      {
		typedef Dune::L2Norm< GridPartType > NormType;
		NormType norm( gridPart );
		double myError = norm.distance( gridExactSolution, scheme.solution() );
		double myError2 = myError * myError;
		double error2 = Dune :: MPIManager :: comm().sum( myError2 );
		if( Dune :: MPIManager :: rank() == 0 )
		  std::cout << "\terror: " << std::sqrt(error2) << std::endl;
	      }
	    else
	      {
		std::cout << std::endl;
	      }

	    // test outer convergence or for no refinements
	    if( std::abs( energy - energyPrerefine ) < outerTol || step < 0 )
	      {
		// output final solution
		dataOutput.write( timeProvider );

		return energy;
	      }
	    else
	      {
		// refinement
		if( scheme.mark() )
		  {
		    adaptationManager.adapt();
		    adapter = true;
		    scheme.multEps( epsFact );
		    timeStep *= tauFact;
		  }

		energyPrerefine = energy;
	      }
	  }

	energyOld = energy;
	nextEnergyTime += nextEnergyUpdate;
      }
  }

  return scheme.energy();
}

// main
// ----

int main ( int argc, char **argv )
try
{
  // initialize MPI, if necessary
  Dune::MPIManager::initialize( argc, argv );

  // append overloaded parameters from the command line
  Dune::Parameter::append( argc, argv );

  // append possible given parameter files
  for( int i = 1; i < argc; ++i )
    Dune::Parameter::append( argv[ i ] );

  // append default parameter file
  Dune::Parameter::append( "../data/parameter" );

  // type of hierarchical grid
  //typedef Dune :: AlbertaGrid< 2 , 2 > GridType;
  typedef Dune :: GridSelector :: GridType  HGridType ;

  // create grid from DGF file
  const std::string gridkey = Dune::IOInterface::defaultGridKey( HGridType::dimension );
  const std::string gridfile = Dune::Parameter::getValue< std::string >( gridkey );

  // the method rank and size from MPIManager are static
  if( Dune::MPIManager::rank() == 0 )
    std::cout << "Loading macro grid: " << gridfile << std::endl;

  // construct macro using the DGF Parser
  Dune::GridPtr< HGridType > gridPtr( gridfile );
  HGridType& grid = *gridPtr ;

  // do initial load balance
  grid.loadBalance();

  // setup EOC loop
  const int repeats = Dune::Parameter::getValue< int >( "partition.repeats", 0 );

  // initial grid refinement
  const int level = Dune::Parameter::getValue< int >( "partition.level" );

  // number of global refinements to bisect grid width
  const int refineStepsForHalf = Dune::DGFGridInfo< HGridType >::refineStepsForHalf();

  // refine grid
  Dune :: GlobalRefine::apply( grid, level * refineStepsForHalf );

  // calculate first step
  double oldError = algorithm( grid, (repeats > 0) ? 0 : -1 );

  for( int step = 1; step <= repeats; ++step )
  {
    // refine globally such that grid with is bisected
    // and all memory is adjusted correctly
    Dune :: GlobalRefine::apply( grid, refineStepsForHalf );

    const double newError = algorithm( grid, step );
    const double eoc = log( oldError / newError ) / M_LN2;
    if( Dune::MPIManager::rank() == 0 )
    {
      std::cout << "Error: " << newError << std::endl;
      std::cout << "EOC( " << step << " ) = " << eoc << std::endl;
    }
    oldError = newError;
  }

  return 0;
}
catch( const Dune::Exception &exception )
{
  std::cerr << "Error: " << exception << std::endl;
  return 1;
}

