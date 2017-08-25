/*
 * Connect in minimum time a starting position to a final state.
 */

#include <acado_toolkit.hpp>
#include "pycado/utils.hpp"

int main(int argc, const char ** argv )
{
  USING_NAMESPACE_ACADO;
  Timer timer;

  /* --- OPTIONS ----------------------------------------------------------------------------- */
  /* --- OPTIONS ----------------------------------------------------------------------------- */
  /* --- OPTIONS ----------------------------------------------------------------------------- */
  OptionsOCP opts; opts.parse(argc,argv);
  opts.NQ = 2; opts.NV = 2;
  assert( opts.friction().size() == 2);
  assert( opts.umax()    .size() == 2);

  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */

  /// Pendulum hyperparameters

  const double p     =   .5 ;    // lever arm
  const double m     =  3.  ;    // mass
  const double a     =  2.  ;    // armature
  const double g     =  9.81;    // gravity constant
  const double Kf0   = opts.friction()[0], Kf1 = opts.friction()[1];     // friction coeff
  const double DT    = opts.T()/opts.steps(); // integration time
  const double umax0 = opts.umax()[0], umax1 = opts.umax()[1];

  DifferentialState        q0,q1,vq0,vq1;
  Control                  u0,u1  ;       // the control input u = a + b*t + .5*c*t*t
  Parameter                T;
  DifferentialEquation     f( 0.0, T );   // the differential equation

  //  --- SETUP OCP -----------------------
  OCP ocp( 0.0, T, opts.steps() );                        // time horizon of the OCP: [0,T]
  ocp.minimizeMayerTerm( T );

  IntermediateState nM = (16*a*a + 16*a*m*p*p*cos(q1) + 28*a*m*p*p 
                          + 4*m*m*p*p*p*p*sin(q1)*sin(q1) + m*m*p*p*p*p);
  IntermediateState 
    Mi00 = (  16*a + 4*m*p*p)/nM,
    Mi01 = ( -4*m*p*p*(2*cos(q1) + 1))/nM,
    Mi11 = (16*a + 8*m*p*p*(2*cos(q1) + 3))/nM,
    b0    = -m*p*(3*g*sin(q0) + g*sin(q0 + q1) + 2*p*vq0*vq1*sin(q1) + p*vq1*vq1*sin(q1))/2,
    b1    =    m*p*(-g*sin(q0 + q1) + p*vq0*vq0*sin(q1))/2;
 
  IntermediateState 
    tau0 = u0 - Kf0*vq0 - b0,
    tau1 = u1 - Kf1*vq1 - b1;

  f << dot(q0)  == vq0;
  f << dot(vq0) == Mi00*tau0 + Mi01*tau1;
  f << dot(q1)  == vq1;
  f << dot(vq1) == Mi01*tau0 + Mi11*tau1;

  ocp.subjectTo( f );

  ocp.subjectTo( AT_START,  q0  ==  opts.configInit()[0] );
  ocp.subjectTo( AT_START,  vq0 ==  opts.velInit   ()[0] );
  ocp.subjectTo( AT_START,  q1  ==  opts.configInit()[1] );
  ocp.subjectTo( AT_START,  vq1 ==  opts.velInit   ()[1] );

  ocp.subjectTo( AT_END,    q0  ==  opts.configFinal()[0] );
  ocp.subjectTo( AT_END,    vq0 ==  opts.velFinal()   [0] );
  ocp.subjectTo( AT_END,    q1  ==  opts.configFinal()[1] );
  ocp.subjectTo( AT_END,    vq1 ==  opts.velFinal()   [1] );

  ocp.subjectTo( -opts.umax()[0]  <= u0 <=  opts.umax()[0]   );
  ocp.subjectTo( -opts.umax()[1]  <= u1 <=  opts.umax()[1]   );

  ocp.subjectTo(  opts.Tmin()     <= T  <= opts.Tmax()       );

  ocp.subjectTo( -M_PI  <= q1 <=  M_PI   );

  //  --- SETUP SOLVER --------------------

  OptimizationAlgorithm algorithm(ocp);

  setupPlots(algorithm,opts,q0,q1,u0,u1);
  initControlAndState(algorithm,opts);
  initHorizon(algorithm,opts);
  initAlgorithmStandardParameters(algorithm,opts);

  returnValue retval = algorithm.solve();

  outputControlAndState(algorithm,opts);
  outputParameters(algorithm,opts);

  //  --- RETURN --------------------------
  std::cout << "###### Return["<<int(retval)<<"] JobID=" << opts.jobid() << timer << std::endl;
  return (int)retval;
}
