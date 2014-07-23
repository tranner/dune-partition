#ifndef PARTITION_FEMSCHEME_HH
#define PARTITION_FEMSCHEME_HH

// include parameter handling
#include <dune/fem/io/parameter.hh>

// local includes
#include "heatscheme.hh"

// PartitionScheme
//------------------

template < class ImplicitModel, class ExplicitModel >
struct PartitionScheme : public HeatScheme<ImplicitModel, ExplicitModel>
{
  typedef HeatScheme<ImplicitModel,ExplicitModel> BaseType;
  typedef typename BaseType::GridType GridType;
  typedef typename BaseType::GridPartType GridPartType;
  typedef typename BaseType::ModelType ImplicitModelType;
  typedef typename BaseType::ExplicitModelType ExplicitModelType;
  typedef typename BaseType::FunctionSpaceType FunctionSpaceType;
  typedef typename BaseType::DiscreteFunctionSpaceType DiscreteFunctionSpaceType;
  typedef typename BaseType::DiscreteFunctionType DiscreteFunctionType;

  typedef typename DiscreteFunctionSpaceType::IteratorType IteratorType;
  typedef typename IteratorType::Entity EntityType;
  typedef typename EntityType::Geometry GeometryType;

  typedef typename DiscreteFunctionType :: LocalFunctionType LocalFunctionType;
  typedef typename DiscreteFunctionType :: RangeType RangeType;
  typedef typename DiscreteFunctionType :: JacobianRangeType JacobianRangeType;

  typedef Dune::Fem::CachingQuadrature< GridPartType, 0 > QuadratureType;

  PartitionScheme( GridPartType &gridPart,
		   const ImplicitModelType& implicitModel,
		   const ExplicitModelType& explicitModel )
    : BaseType(gridPart, implicitModel, explicitModel),
      eps_( Dune::Parameter::getValue< double >( "partition.epsilon", 0.1 ) ),
      maxLevel_( Dune::Parameter::getValue< int >( "partition.maxlevel", 10 ) )
  {}

  void solveODE()
  {
    const double tau = implicitModel_.timeProvider().deltaT();
    const double eps2 = eps_*eps_;

    const unsigned int M = (*solution_.block(0)).size();
    const unsigned int size = solution_.size() / M;
    // for each node
    for( unsigned int i = 0; i < size; ++i )
      {
	// find solution
	RangeType &ux = *solution_.block( i );
	double myNorm2 = ux.two_norm2();
	const double norm2 = Dune::MPIManager::comm().sum( myNorm2 );

	// for each partition
	for( unsigned int j = 0; j < ux.size(); ++j )
	  {
	    // update solution at each component
	    ux[j] *= exp( - 2.0 * tau / eps2 * ( norm2 - ux[j]*ux[j] ) );
	  }
      }
  }

  void normalise()
  {
    // calculate l2 norm
    typedef Dune :: Fem :: L2Norm< GridPartType > NormType;
    NormType l2Norm( gridPart_ );
    const double norm = l2Norm.norm( solution_ );
    assert( norm > 1.0e-8 );

    // rescale solution_
    solution_ /= norm;
  }

  double energy() const
  {
    double E2;
    return energy( E2 );
  }
  
  double energy( double &E2 ) const
  {
    double E1 = 0.0;
    E2 = 0.0;

    const int rank = Dune :: MPIManager :: rank();
    const int size = Dune :: MPIManager :: size();

    const IteratorType end = discreteSpace_.end();
    for( IteratorType it = discreteSpace_.begin(); it != end; ++it )
      {
	const EntityType &entity = *it;
	const GeometryType &geometry = entity.geometry();

	const LocalFunctionType uLocal = solution_.localFunction( entity );

	QuadratureType quadrature( entity, 4*discreteSpace_.order() );
	const size_t nop = quadrature.nop();
	for( size_t pt = 0; pt < nop; ++pt )
	  {
	    const typename QuadratureType :: CoordinateType &x = quadrature.point( pt );
	    const double weight = quadrature.weight( pt ) * geometry.integrationElement( x );

	    JacobianRangeType dux;
	    uLocal.jacobian( quadrature[ pt ], dux );

	    const double F = Feps( entity, quadrature[ pt ] );
	    // do integration
	    E1 += dux.frobenius_norm2() * weight;
	    E2 += F * weight;
	  }
      }

    // communicate to root process
    double *E1all;
    if( rank == 0 )
      E1all = new double [ size ];

    Dune :: MPIManager :: comm().gather( &E1, E1all, 1, 0 );
    Dune :: MPIManager :: comm().broadcast( &E2, 1, 0 );

    if( rank == 0 )
      {
	for( int i = 0; i < size; ++i )
	  {
	    std::cout << "    lambda[ " << i << " ]: " << E1all[i] << std::endl;
	  }
	std::cout << "    Seps: " << E2 << std::endl;

	delete [] E1all;
      }

    return Dune::MPIManager::comm().sum( E1 ) + E2;
  }

  template< class Point >
  double Feps( const EntityType &entity, const Point pt ) const
  {
    // mpi information
    const int rank = Dune :: MPIManager :: rank();
    const int size = Dune :: MPIManager :: size();

    // find local function
    LocalFunctionType uLocal = solution_.localFunction( entity );

    // evaluation local function as pt
    RangeType ux;
    uLocal.evaluate( pt, ux );

    // find total size
    unsigned int m = ux.size();
    const unsigned int M = Dune :: MPIManager :: comm().sum( m );

    // store ux in pointer
    double *ux_ = new double[ m ];
    for( unsigned int j = 0; j < m; ++j )
      {
	ux_[j] = ux[j];
      }

    // add storage on root processer
    double *uxAll = NULL;
    if( rank == 0 )
      {
	uxAll = new double[ M ];
      }
    Dune :: MPIManager :: comm().gather( ux_, uxAll, m, 0 );

    // compute sum of squares
    double ret = 0.0;
    if( rank == 0 )
      {
	for( unsigned int i = 0; i < M; ++i )
	  for( unsigned int j = 0; j < i; ++j )
	    ret += uxAll[i] * uxAll[i] * uxAll[j] * uxAll[j];
      }

    // free memory
    delete [] ux_;
    if( rank == 0 )
      {
	delete [] uxAll;
      }

    // scale return value
    ret *= 2.0  / (eps_*eps_);

    // communicate
    Dune :: MPIManager :: comm().broadcast( &ret, 1, 0 );

    return ret;
  }

  bool mark()
  {
    unsigned int refCount = 0;
    unsigned int count = 0;

    const IteratorType end = discreteSpace_.end();
    for( IteratorType it = discreteSpace_.begin(); it != end; ++it )
      {
	const EntityType &entity = *it;
	count++;

#if ADAPTIVE
	const GeometryType &geometry = entity.geometry();
	const double F = Feps( entity, geometry.local( geometry.center() ) );
	if( F > eps_ )
	  {
	    gridPart_.grid().mark( 2, entity );
	    refCount++;
	  }
#else
	gridPart_.grid().mark( 2, entity );
	refCount++;
#endif
      }

    std::cout << "[" << Dune :: MPIManager :: rank() << "] marked for refined: " << refCount << " of " << count << std::endl;
    return (bool) refCount;
  }

  const double epsilon() const
  {
    return eps_;
  }

  void multEps( const double lambda )
  {
    eps_ *= lambda;
  }

private:
  using BaseType::implicitModel_;
  using BaseType::discreteSpace_;
  using BaseType::solution_;
  using BaseType::gridPart_;

  double eps_;
  const int maxLevel_;
};

#endif // end #if PARTITION_FEMSCHEME_HH
