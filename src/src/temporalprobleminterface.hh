#ifndef HEAT_PROBLEMINTERFACE_HH
#define HEAT_PROBLEMINTERFACE_HH

#include <cassert>
#include <cmath>

#include <dune/fem/solver/timeprovider.hh>

#include "probleminterface.hh"

/** \brief problem interface class for time dependent problem descriptions, i.e. right hand side,
 *         boudnary data, and, if exsistent, an exact solution. A routine time() is
 *         provided. 
 */
template <class FunctionSpace>
class TemporalProblemInterface : public ProblemInterface<FunctionSpace>
{
public:
  typedef Dune::Fem::TimeProviderBase  TimeProviderType ;

  //! constructor taking time provider
  TemporalProblemInterface( const TimeProviderType &timeProvider ) 
    : timeProvider_(timeProvider)
  {
  }

  //! return current simulation time 
  double time() const
  {
    return timeProvider_.time();
  }

  //! return current time step size (\delta t)
  double deltaT() const 
  {
    return timeProvider_.deltaT() ;
  }

  //! return reference to Problem's time provider 
  const TimeProviderType & timeProvider() const 
  {
    return timeProvider_;
  }

protected:
  const TimeProviderType &timeProvider_;
};
#endif // #ifndef HEAT_PROBLEMINTERFACE_HH

