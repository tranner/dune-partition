#ifndef HEAT_FEMSCHEME_HH
#define HEAT_FEMSCHEME_HH

// iostream includes
#include <iostream>

// include grid part
#include <dune/fem/gridpart/adaptiveleafgridpart.hh>

// adaptation ...
#include <dune/fem/function/adaptivefunction.hh>
#include <dune/fem/space/common/adaptmanager.hh>

// include discrete function
#include <dune/fem/function/blockvectorfunction.hh>

// include linear operators
#include <dune/fem/operator/matrix/spmatrix.hh>
#include <dune/fem/solver/diagonalpreconditioner.hh>

#include <dune/fem/solver/istlsolver.hh>
#include <dune/fem/solver/inverseoperators.hh>

// lagrange interpolation
#include <dune/fem/operator/lagrangeinterpolation.hh>

/*********************************************************/

// include norms
#include <dune/fem/misc/l2norm.hh>
#include <dune/fem/misc/h1norm.hh>

// include parameter handling
#include <dune/fem/io/parameter.hh>

// include output
#include <dune/fem/io/file/dataoutput.hh>

// local includes
#include "femscheme.hh"

// HeatScheme
//------------------

template < class ImplicitModel, class ExplicitModel >
struct HeatScheme : public FemScheme<ImplicitModel>
{
  typedef FemScheme<ImplicitModel> BaseType;
  typedef typename BaseType::GridType GridType;
  typedef typename BaseType::GridPartType GridPartType;
  typedef typename BaseType::ModelType ImplicitModelType;
  typedef ExplicitModel ExplicitModelType;
  typedef typename BaseType::FunctionSpaceType FunctionSpaceType;
  typedef typename BaseType::DiscreteFunctionSpaceType DiscreteFunctionSpaceType;
  typedef typename BaseType::DiscreteFunctionType DiscreteFunctionType;
  HeatScheme( GridPartType &gridPart,
              const ImplicitModelType& implicitModel,
              const ExplicitModelType& explicitModel )
  : BaseType(gridPart, implicitModel),
    explicitModel_(explicitModel),
    explicitOperator_( explicitModel_, discreteSpace_ )
  {
  }

  void prepare()
  {
    // apply constraints, e.g. Dirichlet contraints, to the solution
    explicitOperator_.prepare( explicitModel_.dirichletBoundary(), solution_ );
    // apply explicit operator and also setup right hand side
    explicitOperator_( solution_, rhs_ );
    // apply constraints, e.g. Dirichlet contraints, to the result
    explicitOperator_.prepare( solution_, rhs_ );
  }

  void initialize ()
  {
     Dune::Fem::LagrangeInterpolation
          < typename ExplicitModelType::InitialFunctionType, DiscreteFunctionType > interpolation;
     interpolation( explicitModel_.initialFunction(), solution_ );
  }

protected:
  using BaseType::gridPart_;
  using BaseType::discreteSpace_;
  using BaseType::solution_;
  using BaseType::implicitModel_;
  using BaseType::rhs_;

private:
  const ExplicitModelType &explicitModel_;
  typename BaseType::EllipticOperatorType explicitOperator_; // the operator for the rhs
};

#endif // end #if HEAT_FEMSCHEME_HH
