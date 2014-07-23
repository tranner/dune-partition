#ifndef PARTITION_PROBLEMS_HH
#define PARTITION_PROBLEMS_HH

#include <cassert>
#include <cmath>

#include "temporalprobleminterface.hh"

template <class FunctionSpace>
class PartitionProblem : public TemporalProblemInterface < FunctionSpace >
{
  typedef TemporalProblemInterface < FunctionSpace >  BaseType;
public:
  typedef typename BaseType :: RangeType            RangeType;
  typedef typename BaseType :: DomainType           DomainType;
  typedef typename BaseType :: JacobianRangeType    JacobianRangeType;
  typedef typename BaseType :: DiffusionTensorType  DiffusionTensorType;

  enum { dimRange  = BaseType :: dimRange };
  enum { dimDomain = BaseType :: dimDomain };

  // get time function from base class
  using BaseType :: time ;

  PartitionProblem( const Dune::TimeProviderBase &timeProvider )
    : BaseType( timeProvider ),
      useExactInitialCondition_( Dune :: Parameter :: getValue < bool > ( "partition.useexactinitialcondition", true ) )
  {}

  //! the exact solution
  virtual void u(const DomainType& x,
                 RangeType& phi) const
  {
    phi = RangeType(0);

    const int rank = Dune::MPIManager::rank();
    const int size = Dune::MPIManager::size();

    if( size == 2 && useExactInitialCondition_ )
      {
	const double N = sqrt( 2.0 * M_PI / 3 );
	
	if( x[2] > 0.0 && rank == 0 )
	  phi = x[2] / N;
	else if( x[2] < 0.0 && rank == 1 )
	  phi = -x[2] / N;
      }
    else if( size == 3 && useExactInitialCondition_ )
      {
	const double N = sqrt( 1.0 / 8.0 ) * M_PI;
	
	const double r = x.two_norm();
	const double t = acos( x[2] / r );
	const double p = atan2( x[1], x[0] );
	
	if( p > 0.0 && p < 2.0 * M_PI / 3.0 && rank == 0 )
	  phi = sin( 3.0 * p / 2.0 ) * std::pow( sin( t ), 3.0 / 2.0 ) / N;
	else if( p > - 2.0 * M_PI / 3.0 && p < 0.0 && rank == 1 )
	  phi = -sin( 3.0 * p / 2.0 ) * std::pow( sin( t ), 3.0 / 2.0 ) / N;
	else if( std::abs( p ) > 2.0 * M_PI / 3.0 && rank == 2 )
	  phi = sin( 3.0 * std::abs(p) / 2.0 - M_PI ) * std::pow( sin( t ), 3.0 / 2.0 ) / N;
      }
    else
      {
	assert( phi.size() == 1 );

	int j = -1;
	if( rank == 0 )
	  j = rand() % size;
	Dune :: MPIManager :: comm().broadcast( &j, 1, 0 );
	if( j == rank )
	  phi = 1.0;
      }
  }

private:
  const bool useExactInitialCondition_;
};

#endif // #ifndef PARTITION_PROLEMS_HH
