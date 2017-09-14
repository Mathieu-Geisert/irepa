/*
 * Connect in minimum time a starting position to a final state.
 */

#include <acado_toolkit.hpp>
#include "pycado/utils.hpp"

struct OptionQuadcopter : public OptionsOCP
{
  virtual void addExtraOptions()
  {
    desc.add_options()
      ("maxAngle",       po::value<double>()->default_value(1.7),         "Max angles for roll and pitch");
    desc.add_options()
      ("maxAngleSing",   po::value<double>()->default_value(1.4),         "Max angle for the singular axis i.e pitch");
  }

  const double & maxAngle()        { return vm["maxAngle"].as<double>(); }
  const double & maxAngleSing()        { return vm["maxAngleSing"].as<double>(); }
};

int main(int argc, const char ** argv )
{
  USING_NAMESPACE_ACADO;
  Timer timer;

  /* --- OPTIONS ----------------------------------------------------------------------------- */
  /* --- OPTIONS ----------------------------------------------------------------------------- */
  /* --- OPTIONS ----------------------------------------------------------------------------- */
  OptionsBicopter opts; opts.parse(argc,argv);
  opts.NQ = 5; opts.NV = 5; opts.NU = 4;
  assert( opts.friction().size() == 2);
  assert( opts.umax()    .size() == 4);

  opts.displayBoundaryConditions(std::cout);

  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */

  /// Pendulum hyperparameters

  const double l     =  .5 ;    // lever arm
  const double m     = 2.5 ;    // mass
  const double Ixx   = 1.  ;    // roll inertia XX
  const double Iyy   = 1.  ;    // pitch inertia YY
  const double Izz   = 1.6 ;    // yaw inertia ZZ
  const double Ct    = 0.1 ;    // Coeff torque/force of each propeller
  const double g     = 9.81;    // gravity constant
  const double DT    = opts.T()/opts.steps(); // integration time
  const double umax1 = opts.umax()[0], umax2 = opts.umax()[1]; umax3 = opts.umax()[2]; umax4 = opts.umax()[3];


  //WARNING: The angles roll-pitch-yaw are considered in the inverse order of the usual convention RPY
  //         i.e YAW then PITCH then ROLL. This means that ROLL always affect the world's Y axis, no matter the yaw.
  //
  // p, q, r follow the usual convention i.e p is the rotational speed around the local X axis,
  //                                         q is the rotational speed around the local Y axis,
  //                                         r is the rotational speed around the local Z axis,

  DifferentialState        qx, qy, qz, vx, vy, vz, pitch, roll, p, q, r, yaw;
  Control                  f1,f2,f3,f4;
  Parameter                T;
  DifferentialEquation     f( 0.0, T );   // the differential equation

  IntermediateState dyaw = cos(yaw)*tan(pitch)*p+sin(yaw)*tan(pitch)*q+r;
  IntermediateState dpitch = sin(yaw)*p+cos(yaw)*q;
  IntermediateState droll = cos(yaw)/cos(pitch)*p-sin(yaw)/cos(pitch)*q;
  IntermediateState fTot = f1+f2+f3+f4;

  //  --- SETUP OCP -----------------------
  OCP ocp( 0.0, T, opts.steps() );                        // time horizon of the OCP: [0,T]

  ocp.minimizeMayerTerm( T );

  f << dot(qx) == vx;
  f << dot(qy) == vy;
  f << dot(qz) == vz;
  f << dot(vx) == fTot*sin(pitch)/m;                //(f1+f2+f3+f4)*sin(pitch)/m;
  f << dot(vy) == fTot*sin(roll)*cos(pitch)/m;      //(f1+f2+f3+f4)*sin(roll)*cos(pitch)/m;
  f << dot(vz) == fTot*cos(roll)*cos(pitch)/m - g;  //(f1+f2+f3+f4)*cos(roll)*cos(pitch)/m - g;
  f << dot(yaw) == dyaw;                            //-cos(yaw)*tan(pitch)*p+sin(yaw)*tan(pitch)*q+r;
  f << dot(pitch) == dpitch;                        //sin(yaw)*p+cos(yaw)*q;
  f << dot(roll) == droll;                          //cos(yaw)/cos(pitch)*p-sin(yaw)/cos(pitch)*q;
  f << dot(p) == (d*(u1-u2)+(Iyy-Izz)*q*r)/Ixx;
  f << dot(q) == (d*(u4-u3)+(Izz-Ixx)*p*r)/Iyy;
  f << dot(r) == Ct*(u1+u2-u3-u4)/Izz;

  ocp.subjectTo( f );

  //FIXED CONSTRAINTS:
  //Start
  ocp.subjectTo( AT_START,  yaw  ==  0. );
  ocp.subjectTo( AT_START,  dyaw ==  0. );

  //End
  ocp.subjectTo( AT_END,  yaw  ==  0. );
  ocp.subjectTo( AT_END,  dyaw ==  0. );

  //Controls
  ocp.subjectTo( 0.  <= f1 <=  umax1   );
  ocp.subjectTo( 0.  <= f2 <=  umax2   );
  ocp.subjectTo( 0.  <= f3 <=  umax3   );
  ocp.subjectTo( 0.  <= f4 <=  umax4   );

  //Avoid Flipping roll
  ocp.subjectTo( -opts.maxAngle()  <= roll <=  opts.maxAngle()   );

  //Avoid Flipping or Singularity pitch
  if (opts.maxAngle() > opts.maxAngleSing())
  {
      //Avoid Singularity
      ocp.subjectTo( -opts.maxAngleSing()  <= pitch <=  opts.maxAngleSing()  );
  }
  else
  {
      ocp.subjectTo( -opts.maxAngle()  <= pitch <=  opts.maxAngle()   );
  }

  //Time boundaries
  ocp.subjectTo( opts.Tmin()  <= T  <= opts.Tmax()  );

  //PRM CONSTRAINTS:
  //Start
  ocp.subjectTo( AT_START,  qx  ==  opts.configInit ()[0] );
  ocp.subjectTo( AT_START,  vx  ==  opts.velInit    ()[0] );

  ocp.subjectTo( AT_START,  qy  ==  opts.configInit ()[1] );
  ocp.subjectTo( AT_START,  vy  ==  opts.velInit    ()[1] );

  ocp.subjectTo( AT_START,  qz  ==  opts.configInit ()[2] );
  ocp.subjectTo( AT_START,  vz  ==  opts.velInit    ()[2] );

  ocp.subjectTo( AT_START,  roll ==  opts.configInit ()[3] );
  ocp.subjectTo( AT_START, pitch ==  opts.configInit ()[4] );
//  ocp.subjectTo( AT_START,  yaw ==  opts.configInit ()[5] );

  ocp.subjectTo( AT_START,dpitch ==  opts.velInit    ()[3] );
  ocp.subjectTo( AT_START, droll ==  opts.velInit    ()[4] );
//  ocp.subjectTo( AT_START,  dyaw ==  opts.configInit ()[5] );

//  ocp.subjectTo( AT_START,  p ==  opts.velInit    ()[3] );
//  ocp.subjectTo( AT_START,  q ==  opts.velInit    ()[4] );
//  ocp.subjectTo( AT_START,  r ==  opts.velInit    ()[5] );

  //End
  ocp.subjectTo( AT_END  ,  qx  ==  opts.configFinal()[0] );
  ocp.subjectTo( AT_END  ,  vx  ==  opts.velFinal   ()[0] );

  ocp.subjectTo( AT_END  ,  qy  ==  opts.configFinal()[1] );
  ocp.subjectTo( AT_END  ,  vy  ==  opts.velFinal   ()[1] );

  ocp.subjectTo( AT_END  ,  qz  ==  opts.configFinal()[2] );
  ocp.subjectTo( AT_END  ,  vz  ==  opts.velFinal   ()[2] );

  ocp.subjectTo( AT_END,   roll ==  opts.configFinal ()[3] );
  ocp.subjectTo( AT_END,  pitch ==  opts.configFinal ()[4] );
//  ocp.subjectTo( AT_END,  yaw ==  opts.configFinal ()[5] );

  ocp.subjectTo( AT_END, dpitch ==  opts.velFinal    ()[3] );
  ocp.subjectTo( AT_END,  droll ==  opts.velFinal    ()[4] );
//  ocp.subjectTo( AT_END,  dyaw ==  opts.configFinal ()[5] );

//  ocp.subjectTo( AT_END,  p ==  opts.velFinal    ()[3] );
//  ocp.subjectTo( AT_END,  q ==  opts.velFinal    ()[4] );
//  ocp.subjectTo( AT_END,  r ==  opts.velFinal    ()[5] );


  //  --- SETUP SOLVER --------------------

  OptimizationAlgorithm algorithm(ocp);

  const std::vector<std::string> plotNames = {"X", "Y", "Z", "roll", "pitch"};
  std::vector<ACADO::Expression> plotExpr = {qx, qy, qz, roll, pitch};

  setupPlots(algorithm,opts,plotExpr,plotNames);
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

