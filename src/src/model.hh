#ifndef ELLIPTC_MODEL_HH
#define ELLIPTC_MODEL_HH

#include <cassert>
#include <cmath>

#include <dune/fem/solver/timeprovider.hh>
#include <dune/fem/io/parameter.hh>

#include "probleminterface.hh"

// DiffusionModel
// --------------

template< class FunctionSpace, class GridPart >
struct DiffusionModel
{
  typedef FunctionSpace FunctionSpaceType;
  typedef GridPart GridPartType;

  typedef typename FunctionSpaceType::DomainType DomainType;
  typedef typename FunctionSpaceType::RangeType RangeType;
  typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;

  typedef typename FunctionSpaceType::DomainFieldType DomainFieldType;
  typedef typename FunctionSpaceType::RangeFieldType RangeFieldType;

  typedef ProblemInterface< FunctionSpaceType > ProblemType ;

protected:
  enum FunctionId { rhs, bnd };
  template <FunctionId id>
  class FunctionWrapper;
public:
  typedef Dune::Fem::GridFunctionAdapter< FunctionWrapper<rhs>, GridPartType > RightHandSideType;
  typedef Dune::Fem::GridFunctionAdapter< FunctionWrapper<bnd>, GridPartType > DirichletBoundaryType;

  //! constructor taking problem reference 
  DiffusionModel( const ProblemType& problem, const GridPart &gridPart )
    : problem_( problem ),
      gridPart_(gridPart),
      rhs_(problem_),
      bnd_(problem_)
  {
  }

  template< class Entity, class Point >
  void source ( const Entity &entity, 
                const Point &x,
                const RangeType &value, 
                RangeType &flux ) const
  {
    linSource( value, entity, x, value, flux );
  }

  // the linearization of the source function
  template< class Entity, class Point >
  void linSource ( const RangeType& uBar, 
                   const Entity &entity, 
                   const Point &x,
                   const RangeType &value, 
                   RangeType &flux ) const
  {
    const DomainType xGlobal = entity.geometry().global( coordinate( x ) );
    RangeType m;
    problem_.m(xGlobal,m);
    for (unsigned int i=0;i<flux.size();++i)
      flux[i] = m[i]*value[i];
  }
  //! return the diffusive flux 
  template< class Entity, class Point >
  void diffusiveFlux ( const Entity &entity, 
                       const Point &x,
                       const RangeType &value,
                       const JacobianRangeType &gradient,
                       JacobianRangeType &flux ) const
  {
    linDiffusiveFlux( value, gradient, entity, x, value, gradient, flux );
  }
  // linearization of diffusiveFlux
  template< class Entity, class Point >
  void linDiffusiveFlux ( const RangeType& uBar, 
                          const JacobianRangeType& gradientBar,
                          const Entity &entity, 
                          const Point &x,
                          const RangeType &value,
                          const JacobianRangeType &gradient,
                          JacobianRangeType &flux ) const
  {
    // the flux is simply the identity 
    flux = gradient;
  }

  //! exact some methods from the problem class
  bool hasDirichletBoundary () const 
  {
    return problem_.hasDirichletBoundary() ;
  }

  //! return true if given point belongs to the Dirichlet boundary (default is true)
  bool isDirichletPoint( const DomainType& x ) const 
  {
    return problem_.isDirichletPoint(x) ;
  }

  template< class Entity, class Point >
  void g( const RangeType& uBar, 
          const Entity &entity, 
          const Point &x,
          RangeType &u ) const
  {
    const DomainType xGlobal = entity.geometry().global( coordinate( x ) );
    problem_.g( xGlobal, u );
  }

  // return Fem :: Function for Dirichlet boundary values 
  DirichletBoundaryType dirichletBoundary( ) const 
  {
    return DirichletBoundaryType( "boundary function", bnd_, gridPart_, 5 );  
  }

  // return Fem :: Function for right hand side 
  RightHandSideType rightHandSide(  ) const 
  {
    return RightHandSideType( "right hand side", rhs_, gridPart_, 5 );  
  }
   
protected:
  template <FunctionId id>
  class FunctionWrapper : public Dune::Fem::Function< FunctionSpaceType, FunctionWrapper< id > >
  {
    const ProblemInterface<FunctionSpaceType>& impl_;
    public:   
    FunctionWrapper( const ProblemInterface<FunctionSpaceType>& impl )
    : impl_( impl ) {}
 
    //! evaluate function 
    void evaluate( const DomainType& x, RangeType& ret ) const 
    {
      if( id == rhs ) 
      {
        // call right hand side of implementation 
        impl_.f( x, ret );
      }
      else if( id == bnd ) 
      {
        // call dirichlet boudary data of implementation 
        impl_.g( x, ret );
      }
      else 
      {
        DUNE_THROW(Dune::NotImplemented,"FunctionId not implemented"); 
      }
    }
  };
   
  const ProblemType& problem_;
  const GridPart &gridPart_;
  FunctionWrapper<rhs> rhs_;
  FunctionWrapper<bnd> bnd_;
};

#endif // #ifndef ELLIPTC_MODEL_HH
