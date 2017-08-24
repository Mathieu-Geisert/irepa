/*
 * Connect in minimum time a starting position to a final state.
 */

#include <acado_toolkit.hpp>
#include "pycado/utils.hpp"

struct OptionsBicopter : public OptionsOCP
{
  virtual void addExtraOptions()
  {
    desc.add_options()
      ("thetamax",       po::value<double     >()->default_value(1.7),         "Theta max");
  }

  const double & thetaMax()        { return vm["thetamax"].as<double>(); }
};

int main(int argc, const char ** argv )
{
  USING_NAMESPACE_ACADO;
  Timer timer;

  /* --- OPTIONS ----------------------------------------------------------------------------- */
  /* --- OPTIONS ----------------------------------------------------------------------------- */
  /* --- OPTIONS ----------------------------------------------------------------------------- */
  OptionsOCP opts; opts.parse(argc,argv);
  opts.NQ = 3; opts.NV = 3; opts.NU = 2;
  assert( opts.friction().size() == 2);
  assert( opts.umax()    .size() == 2);

  opts.displayBoundaryConditions(std::cout);

  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */

  /// Pendulum hyperparameters

  const double l     =   .5 ;    // lever arm
  const double m     =  3.  ;    // mass
  const double I     =  1.  ;
  const double g     =  9.81;    // gravity constant
  const double DT    = opts.T()/opts.steps(); // integration time
  const double umax0 = opts.umax()[0], umax1 = opts.umax()[1];

  //DifferentialState        qx,qz,qth,vx,vz,vth;
  DifferentialState        qx,qz,vx,vz;
  Control                  fp,fn  ;
  Parameter                T;
  DifferentialEquation     f( 0.0, T );   // the differential equation

  //  --- SETUP OCP -----------------------
  OCP ocp( 0.0, T, opts.steps() );                        // time horizon of the OCP: [0,T]
  ocp.minimizeMayerTerm( T );

  f << dot(qx)   == vx;
  f << dot(vx)   == fp;//-1/m*(fp+fn)*sin(qth);
  f << dot(qz)   == vz;
  f << dot(vz)   == fn;//+1/m*(fp+fn)*cos(qth)-g;
  //f << dot(qth)  == vth;
  //f << dot(vth)  == 0;//l/I*(fp-fn);

  ocp.subjectTo( f );

  ocp.subjectTo( AT_START,  qx  ==  opts.configInit ()[0] );
  ocp.subjectTo( AT_START,  vx  ==  opts.velInit    ()[0] );
  ocp.subjectTo( AT_START,  qz  ==  opts.configInit ()[1] );
  ocp.subjectTo( AT_START,  vz  ==  opts.velInit    ()[1] );
  // ocp.subjectTo( AT_START,  qth ==  opts.configInit ()[2] );
  // ocp.subjectTo( AT_START,  vth ==  opts.velInit    ()[2] );

  ocp.subjectTo( AT_END  ,  qx  ==  opts.configFinal()[0] );
  ocp.subjectTo( AT_END  ,  vx  ==  opts.velFinal   ()[0] );
  ocp.subjectTo( AT_END  ,  qz  ==  opts.configFinal()[1] );
  ocp.subjectTo( AT_END  ,  vz  ==  opts.velFinal   ()[1] );
  // ocp.subjectTo( AT_END  ,  qth ==  opts.configFinal()[2] );
  // ocp.subjectTo( AT_END  ,  vth ==  opts.velFinal   ()[2] );

  ocp.subjectTo( 0  <= fp <=  opts.umax()[0]   );
  ocp.subjectTo( 0  <= fn <=  opts.umax()[1]   );

  ocp.subjectTo( opts.Tmin()  <= T  <= opts.Tmax()  );

  //ocp.subjectTo( -M_PI  <= qth <=  M_PI   );

  //  --- SETUP SOLVER --------------------

  OptimizationAlgorithm algorithm(ocp);

  setupPlots(algorithm,opts,qx,qz,fn,fp);
  initControlAndState(algorithm,opts);
  initHorizon(algorithm,opts);
  initAlgorithmStandardParameters(algorithm,opts);

  returnValue retval = algorithm.solve();

  //outputControlAndState(algorithm,opts);
  //outputParameters(algorithm,opts);

  //  --- RETURN --------------------------
  std::cout << "###### Return["<<int(retval)<<"] JobID=" << opts.jobid() << timer << std::endl;
  return (int)retval;
}
