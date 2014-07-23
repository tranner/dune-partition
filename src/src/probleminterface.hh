#ifndef POISSON_PROBLEMINTERFACE_HH
#define POISSON_PROBLEMINTERFACE_HH

#include <cassert>
#include <cmath>

#include <dune/common/exceptions.hh>
#include <dune/fem/function/common/function.hh>

/** \brief problem interface class for problem descriptions, i.e. right hand side,
 *         boudnary data, and, if exsistent, an exact solution.
 */
template <class FunctionSpace>
class ProblemInterface : public Dune::Fem::Function< FunctionSpace, ProblemInterface<FunctionSpace> >
{
public:  
  // type of function space 
  typedef FunctionSpace  FunctionSpaceType;  

  enum { dimRange  = FunctionSpaceType :: dimRange  };
  enum { dimDomain = FunctionSpaceType :: dimDomain };

  typedef typename FunctionSpaceType :: RangeFieldType   RangeFieldType;

  typedef typename FunctionSpaceType :: RangeType   RangeType;
  typedef typename FunctionSpaceType :: DomainType  DomainType;

  typedef typename FunctionSpaceType :: JacobianRangeType  JacobianRangeType;

  typedef Dune::FieldMatrix< RangeFieldType, dimDomain, dimDomain > DiffusionTensorType; 

  //! the right hand side data (default = 0)
  virtual void f(const DomainType& x, 
                 RangeType& value) const
  {
    value = 0; 
  }
  
  //! mass coefficient (default = 0)
  virtual void m(const DomainType& x, RangeType &m) const
  {
    m = RangeType(0);
  }

  //! the exact solution (default = 0)
  virtual void u(const DomainType& x, 
                 RangeType& value) const 
  {
    value = 0; 
  }

  //! the jacobian of the exact solution (default = 0)
  virtual void uJacobian(const DomainType& x, 
                         JacobianRangeType& value) const 
  {
    value = 0; 
  }

  //! diffusion coefficient (default = Id)
  virtual void D(const DomainType& x, DiffusionTensorType& D ) const 
  {
    // set to identity by default
    D = 0;
    for( int i=0; i<D.rows; ++i ) 
      D[ i ][ i ] = 1;
  }

  //! return true if Dirichlet boundary is present (default is true)
  virtual bool hasDirichletBoundary () const 
  {
    return false ;
  }

  //! return true if given point belongs to the Dirichlet boundary (default is true)
  virtual bool isDirichletPoint( const DomainType& x ) const 
  {
    return false ;
  }

  //! the Dirichlet boundary data (default calls u)
  virtual void g(const DomainType& x, 
                 RangeType& value) const 
  {
    u( x, value );
  }

  //! make this into a fem function for the exact solution
  void evaluate( const DomainType& x, RangeType& ret ) const 
  {
    // call exact solution of implementation 
    u( x, ret );
  }
  //! also need the jacobian of the exact solution
  void jacobian( const DomainType& x, JacobianRangeType& jac ) const 
  {
    uJacobian( x, jac );
  }
};

#endif // #ifndef ELLIPTC_PROBLEMINTERFACE_HH

